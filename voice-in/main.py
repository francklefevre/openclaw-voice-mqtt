#!/usr/bin/env python3
"""
voice-in: Capture audio from microphone, detect speech via VAD,
transcribe with ASR (Whisper), and publish text to MQTT.
"""

import io
import json
import os
import sys
import wave
import threading
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import sounddevice as sd
import torch
import yaml

# ── Config ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config from config.yaml, falling back to config.example.yaml."""
    for name in ("config.yaml", "config.example.yaml"):
        path = Path(__file__).parent.parent / name
        if path.exists():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            # Expand env vars in string values
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


# ── VAD (Silero) ────────────────────────────────────────────────────────────

class SileroVAD:
    """Voice Activity Detection using Silero VAD."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if an audio chunk contains speech."""
        tensor = torch.from_numpy(audio_chunk).float()
        if tensor.dim() > 1:
            tensor = tensor.mean(dim=1)
        # Silero VAD expects 512 samples at 16kHz (32ms)
        confidence = self.model(tensor, self.sample_rate).item()
        return confidence > 0.5

    def reset(self):
        """Reset VAD state between utterances."""
        self.model.reset_states()


# ── ASR (Whisper API) ───────────────────────────────────────────────────────

class WhisperASR:
    """Automatic Speech Recognition using OpenAI Whisper API."""

    def __init__(self, api_key: str, model: str = "whisper-1", language: str = "fr"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.language = language

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio data to text."""
        # Convert to WAV in memory
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            # Convert float32 [-1, 1] to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        buf.seek(0)
        buf.name = "audio.wav"

        transcript = self.client.audio.transcriptions.create(
            model=self.model,
            file=buf,
            language=self.language,
        )
        return transcript.text.strip()


# ── MQTT Publisher ──────────────────────────────────────────────────────────

class MQTTPublisher:
    """Publish transcribed text to MQTT."""

    def __init__(self, broker: str, topic_prefix: str):
        self.topic_in = f"{topic_prefix}/in"
        self.reply_topic = f"{topic_prefix}/voice/out"

        # Parse broker URL
        broker_clean = broker.replace("mqtt://", "").replace("mqtts://", "")
        host_parts = broker_clean.split(":")
        host = host_parts[0]
        port = int(host_parts[1]) if len(host_parts) > 1 else 1883

        self.client = mqtt.Client(
            client_id=f"voice-in-{os.getpid()}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.connect(host, port)
        self.client.loop_start()
        print(f"MQTT connected to {host}:{port}")
        print(f"  Publishing to: {self.topic_in}")
        print(f"  Reply topic:   {self.reply_topic}")

    def publish(self, text: str):
        """Publish transcribed text to MQTT."""
        payload = json.dumps({
            "text": text,
            "replyTopic": self.reply_topic,
            "sender": "voice-frontend",
        })
        self.client.publish(self.topic_in, payload)
        print(f"  → MQTT: {text}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()


# ── Main Loop ───────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    sample_rate = cfg.get("vad", {}).get("sample_rate", 16000)
    silence_ms = cfg.get("vad", {}).get("silence_threshold_ms", 1500)
    input_device = cfg.get("audio", {}).get("input_device", None)

    print("═══════════════════════════════════════════")
    print("  voice-in 🎙️  — OpenClaw Voice Frontend")
    print("═══════════════════════════════════════════")
    print(f"  Sample rate:      {sample_rate} Hz")
    print(f"  Silence threshold: {silence_ms} ms")
    print()

    # Init components
    print("Loading VAD model...")
    vad = SileroVAD(sample_rate=sample_rate)

    print("Initializing ASR...")
    asr_cfg = cfg.get("asr", {})
    asr = WhisperASR(
        api_key=asr_cfg.get("openai_api_key", ""),
        model=asr_cfg.get("model", "whisper-1"),
        language=asr_cfg.get("language", "fr"),
    )

    print("Connecting to MQTT...")
    mqtt_cfg = cfg.get("mqtt", {})
    publisher = MQTTPublisher(
        broker=mqtt_cfg.get("broker", "mqtt://localhost"),
        topic_prefix=mqtt_cfg.get("topic_prefix", "openclaw"),
    )

    # VAD parameters
    chunk_duration_ms = 32  # Silero VAD expects 32ms chunks at 16kHz
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    silence_chunks = int(silence_ms / chunk_duration_ms)

    # State
    is_speaking = False
    silence_count = 0
    audio_buffer = []

    print()
    print("🎤 Listening... (Ctrl+C to stop)")
    print()

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_speaking, silence_count, audio_buffer

        if status:
            print(f"  ⚠ Audio: {status}", file=sys.stderr)

        audio = indata[:, 0].copy()  # mono

        # Check for speech
        if len(audio) >= chunk_samples:
            speech_detected = vad.is_speech(audio[:chunk_samples])
        else:
            speech_detected = False

        if speech_detected:
            if not is_speaking:
                print("  🗣️  Speech detected")
                is_speaking = True
                silence_count = 0
            audio_buffer.append(audio)
            silence_count = 0
        elif is_speaking:
            audio_buffer.append(audio)
            silence_count += 1

            if silence_count >= silence_chunks:
                # End of utterance — transcribe
                print("  ✋ Silence detected — transcribing...")
                full_audio = np.concatenate(audio_buffer)

                # Transcribe in a separate thread to not block audio
                def do_transcribe(audio_data):
                    try:
                        text = asr.transcribe(audio_data, sample_rate)
                        if text:
                            print(f"  📝 \"{text}\"")
                            publisher.publish(text)
                        else:
                            print("  (empty transcription)")
                    except Exception as e:
                        print(f"  ❌ ASR error: {e}", file=sys.stderr)

                threading.Thread(
                    target=do_transcribe,
                    args=(full_audio,),
                    daemon=True,
                ).start()

                # Reset state
                is_speaking = False
                silence_count = 0
                audio_buffer = []
                vad.reset()

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            device=input_device,
            callback=audio_callback,
        ):
            # Keep main thread alive
            threading.Event().wait()
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        publisher.disconnect()


if __name__ == "__main__":
    main()
