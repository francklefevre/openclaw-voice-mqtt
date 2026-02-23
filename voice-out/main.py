#!/usr/bin/env python3
"""
voice-out: Subscribe to MQTT reply topic, convert text to speech via TTS API,
and play audio on the local speaker.
"""

import io
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import sounddevice as sd
import yaml


# ── Config ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config from config.yaml, falling back to config.example.yaml."""
    for name in ("config.yaml", "config.example.yaml"):
        path = Path(__file__).parent.parent / name
        if path.exists():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            _expand_env(cfg)
            return cfg
    print("ERROR: No config.yaml found. Copy config.example.yaml to config.yaml")
    sys.exit(1)


def _expand_env(obj):
    """Recursively expand ${VAR} references in string values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                obj[k] = os.environ.get(v[2:-1], v)
            else:
                _expand_env(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                obj[i] = os.environ.get(v[2:-1], v)
            else:
                _expand_env(v)


# ── Audio device helper ────────────────────────────────────────────────────

def resolve_audio_device(configured, kind="output"):
    """Resolve audio device: try configured value, fall back to system default.
    kind is 'input' or 'output'."""
    if configured is not None:
        try:
            info = sd.query_devices(configured, kind)
            print(f"  Audio {kind}: #{info['index']} — {info['name']}")
            return configured
        except ValueError:
            print(f"  WARNING: {kind} device {configured!r} not found, falling back to default")
    default = sd.default.device[0 if kind == "input" else 1]
    try:
        info = sd.query_devices(default, kind)
        print(f"  Audio {kind}: #{info['index']} — {info['name']} (default)")
    except Exception:
        print(f"  Audio {kind}: system default (could not query details)")
    return None


# ── TTS (OpenAI) ───────────────────────────────────────────────────────────

class OpenAITTS:
    """Text-to-Speech using OpenAI API."""

    def __init__(self, api_key: str, model: str = "tts-1", voice: str = "nova"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (PCM)."""
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",  # Raw 24kHz 16-bit mono PCM
        )
        return response.read()


# ── Audio Player ────────────────────────────────────────────────────────────

class AudioPlayer:
    """Play PCM audio on the local speaker."""

    def __init__(self, output_device=None):
        self.output_device = output_device
        self.sample_rate = 24000  # OpenAI TTS outputs 24kHz
        self._lock = threading.Lock()

    def play(self, pcm_data: bytes):
        """Play raw PCM audio (24kHz, 16-bit, mono)."""
        with self._lock:
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32767.0
            sd.play(audio, samplerate=self.sample_rate, device=self.output_device)
            sd.wait()


# ── MQTT Subscriber ─────────────────────────────────────────────────────────

class VoiceOutSubscriber:
    """Subscribe to MQTT and play incoming text as speech."""

    def __init__(self, broker: str, topic: str, topic_prefix: str, tts: OpenAITTS, player: AudioPlayer):
        self.topic = topic
        self.mute_topic = f"{topic_prefix}/voice/mute"
        self.tts = tts
        self.player = player

        # Parse broker URL
        broker_clean = broker.replace("mqtt://", "").replace("mqtts://", "")
        host_parts = broker_clean.split(":")
        host = host_parts[0]
        port = int(host_parts[1]) if len(host_parts) > 1 else 1883

        self.client = mqtt.Client(
            client_id=f"voice-out-{os.getpid()}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(host, port)

        print(f"MQTT connected to {host}:{port}")
        print(f"  Subscribed to: {self.topic}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Subscribe on (re)connect."""
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT message."""
        try:
            raw = msg.payload.decode("utf-8")
            try:
                parsed = json.loads(raw)
                text = parsed.get("text", "").strip()
            except json.JSONDecodeError:
                text = raw.strip()

            if not text:
                return

            print(f"  📩 Received: \"{text}\"")
            print(f"  🔊 Synthesizing speech...")

            # TTS + playback in a thread to not block MQTT
            def do_speak(t):
                try:
                    pcm = self.tts.synthesize(t)
                    print(f"  ▶️  Playing ({len(pcm)} bytes)")
                    # Mute voice-in before playback to prevent audio loop
                    self.client.publish(self.mute_topic, json.dumps({"muted": True}))
                    self.player.play(pcm)
                    # Unmute voice-in after playback
                    self.client.publish(self.mute_topic, json.dumps({"muted": False}))
                    print(f"  ✅ Done")
                except Exception as e:
                    # Always unmute on error
                    self.client.publish(self.mute_topic, json.dumps({"muted": False}))
                    print(f"  ❌ TTS/playback error: {e}", file=sys.stderr)

            threading.Thread(target=do_speak, args=(text,), daemon=True).start()

        except Exception as e:
            print(f"  ❌ Message error: {e}", file=sys.stderr)

    def loop_forever(self):
        """Run the MQTT event loop."""
        self.client.loop_forever()

    def disconnect(self):
        self.client.disconnect()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    output_device = cfg.get("audio", {}).get("output_device", None)

    print("═══════════════════════════════════════════")
    print("  voice-out 🔊 — OpenClaw Voice Frontend")
    print("═══════════════════════════════════════════")
    output_device = resolve_audio_device(output_device, "output")
    print()

    # Init TTS
    print("Initializing TTS...")
    tts_cfg = cfg.get("tts", {})
    tts = OpenAITTS(
        api_key=tts_cfg.get("openai_api_key", ""),
        model=tts_cfg.get("model", "tts-1"),
        voice=tts_cfg.get("voice", "nova"),
    )

    # Init player
    player = AudioPlayer(output_device=output_device)

    # Init MQTT
    print("Connecting to MQTT...")
    mqtt_cfg = cfg.get("mqtt", {})
    topic_prefix = mqtt_cfg.get("topic_prefix", "openclaw")
    reply_topic = f"{topic_prefix}/voice/out"

    subscriber = VoiceOutSubscriber(
        broker=mqtt_cfg.get("broker", "mqtt://localhost"),
        topic=reply_topic,
        topic_prefix=topic_prefix,
        tts=tts,
        player=player,
    )

    print()
    print("👂 Waiting for replies... (Ctrl+C to stop)")
    print()

    try:
        subscriber.loop_forever()
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        subscriber.disconnect()


if __name__ == "__main__":
    main()
