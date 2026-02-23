#!/usr/bin/env python3
"""
voice-out: Subscribe to MQTT reply topic, convert text to speech via TTS API,
and play audio on the local speaker. Supports duplex interruption.
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
    """Resolve audio device: try configured value, fall back to PulseAudio
    'pulse' device (if available), then system default."""
    if configured is not None:
        try:
            info = sd.query_devices(configured, kind)
            print(f"  Audio {kind}: #{info['index']} — {info['name']}")
            return configured
        except ValueError:
            print(f"  WARNING: {kind} device {configured!r} not found")
    try:
        info = sd.query_devices("pulse", kind)
        print(f"  Audio {kind}: #{info['index']} — {info['name']} (pulse)")
        return "pulse"
    except ValueError:
        pass
    default = sd.default.device[0 if kind == "input" else 1]
    try:
        info = sd.query_devices(default, kind)
        print(f"  Audio {kind}: #{info['index']} — {info['name']} (default)")
    except Exception:
        print(f"  Audio {kind}: system default (could not query details)")
    return None


# ── TTS (OpenAI) ───────────────────────────────────────────────────────────

class OpenAITTS:
    """Text-to-Speech using OpenAI API with streaming support."""

    def __init__(self, api_key: str, model: str = "tts-1", voice: str = "nova"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    def synthesize_stream(self, text: str):
        """Stream TTS audio as PCM chunks. Yields raw bytes as they arrive."""
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4800):
                if chunk:
                    yield chunk

    def synthesize(self, text: str) -> bytes:
        """Non-streaming fallback: full audio at once."""
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
        )
        return response.read()


# ── Interruptible Audio Player ──────────────────────────────────────────────

class AudioPlayer:
    """Play PCM audio with support for interruption (duplex mode)."""

    def __init__(self, output_device=None):
        self.output_device = output_device
        self.sample_rate = 24000  # OpenAI TTS outputs 24kHz
        self._lock = threading.Lock()
        self.interrupted = threading.Event()
        self.is_playing = threading.Event()

    def interrupt(self):
        """Interrupt current playback immediately."""
        if self.is_playing.is_set():
            self.interrupted.set()
            sd.stop()
            print(f"  ⏹️  Playback interrupted!")

    def play_stream(self, pcm_chunks):
        """Collect streaming PCM chunks then play. Can be interrupted."""
        self.interrupted.clear()
        chunks = []
        total_bytes = 0

        for chunk in pcm_chunks:
            if self.interrupted.is_set():
                print(f"  ⏹️  Stream interrupted during download")
                return total_bytes
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes == len(chunk):
                print(f"  ⚡ First chunk received — buffering stream...")

        if not chunks or self.interrupted.is_set():
            return total_bytes

        print(f"  ▶️  Playing ({total_bytes} bytes)")
        with self._lock:
            all_audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32767.0
            self.is_playing.set()
            sd.play(all_audio, samplerate=self.sample_rate, device=self.output_device)
            # Wait but check for interruption periodically
            while sd.get_stream().active:
                if self.interrupted.is_set():
                    sd.stop()
                    break
                sd.sleep(50)
            self.is_playing.clear()

        return total_bytes

    def play(self, pcm_data: bytes):
        """Play full PCM audio (non-streaming fallback)."""
        self.interrupted.clear()
        with self._lock:
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32767.0
            self.is_playing.set()
            sd.play(audio, samplerate=self.sample_rate, device=self.output_device)
            while sd.get_stream().active:
                if self.interrupted.is_set():
                    sd.stop()
                    break
                sd.sleep(50)
            self.is_playing.clear()


# ── MQTT Subscriber ─────────────────────────────────────────────────────────

class VoiceOutSubscriber:
    """Subscribe to MQTT and play incoming text as speech. Supports duplex interruption."""

    def __init__(self, broker: str, topic: str, topic_prefix: str, tts: OpenAITTS, player: AudioPlayer):
        self.topic = topic
        self.interrupt_topic = f"{topic_prefix}/voice/interrupt"
        self.speaking_topic = f"{topic_prefix}/voice/speaking"
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
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.connect(host, port, keepalive=60)

        print(f"MQTT connected to {host}:{port}")
        print(f"  Reply topic:     {self.topic}")
        print(f"  Interrupt topic: {self.interrupt_topic}")
        print(f"  Speaking topic:  {self.speaking_topic}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Subscribe on (re)connect."""
        print(f"  🟢 MQTT connected (rc={rc})")
        client.subscribe(self.topic)
        client.subscribe(self.interrupt_topic)
        print(f"  📡 Subscribed to {self.topic} + {self.interrupt_topic}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        """Log disconnections."""
        if rc != 0:
            print(f"  🔴 MQTT disconnected unexpectedly (rc={rc}) — will auto-reconnect")
        else:
            print(f"  ⚪ MQTT disconnected cleanly")

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            # Handle interrupt signal
            if msg.topic == self.interrupt_topic:
                print(f"  🛑 Interrupt received!")
                self.player.interrupt()
                return

            # Handle reply text
            raw = msg.payload.decode("utf-8")
            try:
                parsed = json.loads(raw)
                text = parsed.get("text", "").strip()
            except json.JSONDecodeError:
                text = raw.strip()

            if not text:
                return

            print(f"  📩 Received: \"{text}\"")

            # TTS + streaming playback in a thread to not block MQTT
            def do_speak(t):
                try:
                    # Signal that we're speaking
                    self.client.publish(self.speaking_topic, json.dumps({"speaking": True}))
                    print(f"  🎵 Streaming TTS...")
                    pcm_stream = self.tts.synthesize_stream(t)
                    total = self.player.play_stream(pcm_stream)
                    was_interrupted = self.player.interrupted.is_set()
                    # Signal that we're done
                    self.client.publish(self.speaking_topic, json.dumps({"speaking": False}))
                    if was_interrupted:
                        print(f"  ⏹️  Interrupted after {total} bytes")
                    else:
                        print(f"  ✅ Done ({total} bytes streamed)")
                except Exception as e:
                    self.client.publish(self.speaking_topic, json.dumps({"speaking": False}))
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
    print("  Mode: duplex (interruptible)")
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
    print("💡 Say something while I'm speaking to interrupt me!")
    print()

    try:
        subscriber.loop_forever()
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        subscriber.disconnect()


if __name__ == "__main__":
    main()
