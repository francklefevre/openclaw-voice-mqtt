#!/usr/bin/env python3
"""
voice-in: Capture audio from microphone, detect speech via VAD,
transcribe with ASR (Whisper), and publish text to MQTT.

Supports:
- Anti-loop muting (subscribes to voice/mute topic)
- Wake word detection via OpenWakeWord (optional)
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

def resolve_audio_device(configured, kind="input"):
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


# ── VAD (Silero) ────────────────────────────────────────────────────────────

class SileroVAD:
    """Voice Activity Detection using Silero VAD (ONNX Runtime)."""

    def __init__(self, sample_rate: int = 16000):
        import onnxruntime as ort
        self.sample_rate = sample_rate
        model_path = Path(__file__).parent / "models" / "silero_vad.onnx"
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if an audio chunk contains speech."""
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        x = audio_chunk.astype(np.float32).reshape(1, -1)
        out = self.session.run(None, {
            "input": x,
            "sr": np.array(self.sample_rate, dtype=np.int64),
            "state": self._state,
        })
        confidence, self._state = out[0], out[1]
        return float(confidence.flat[0]) > 0.5

    def reset(self):
        """Reset VAD state between utterances."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)


# ── Wake Word (OpenWakeWord) ────────────────────────────────────────────────

class WakeWordDetector:
    """Wake word detection using OpenWakeWord."""

    def __init__(self, model_name: str = "hey_jarvis", sample_rate: int = 16000):
        from openwakeword.model import Model
        self.model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.threshold = 0.5

    def detect(self, audio_chunk: np.ndarray) -> bool:
        """Check if the wake word is detected in the audio chunk.
        OpenWakeWord expects int16 samples at 16kHz."""
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        prediction = self.model.predict(audio_int16)
        # Check all model keys for activation
        for key in prediction:
            if prediction[key] > self.threshold:
                self.model.reset()
                return True
        return False

    def reset(self):
        """Reset detector state."""
        self.model.reset()


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
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
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
    """Publish transcribed text to MQTT and subscribe to mute topic."""

    def __init__(self, broker: str, topic_prefix: str):
        self.topic_in = f"{topic_prefix}/in"
        self.reply_topic = f"{topic_prefix}/voice/out"
        self.mute_topic = f"{topic_prefix}/voice/mute"
        self.muted = False
        self._mute_lock = threading.Lock()

        # Parse broker URL
        broker_clean = broker.replace("mqtt://", "").replace("mqtts://", "")
        host_parts = broker_clean.split(":")
        host = host_parts[0]
        port = int(host_parts[1]) if len(host_parts) > 1 else 1883

        self.client = mqtt.Client(
            client_id=f"voice-in-{os.getpid()}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(host, port)
        self.client.loop_start()
        print(f"MQTT connected to {host}:{port}")
        print(f"  Publishing to: {self.topic_in}")
        print(f"  Reply topic:   {self.reply_topic}")
        print(f"  Mute topic:    {self.mute_topic}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Subscribe to mute topic on (re)connect."""
        client.subscribe(self.mute_topic)

    def _on_message(self, client, userdata, msg):
        """Handle mute/unmute messages."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            muted = payload.get("muted", False)
            with self._mute_lock:
                self.muted = muted
            state = "🔇 MUTED" if muted else "🔊 UNMUTED"
            print(f"  {state} (anti-loop)")
        except Exception as e:
            print(f"  ⚠ Mute message error: {e}", file=sys.stderr)

    @property
    def is_muted(self) -> bool:
        with self._mute_lock:
            return self.muted

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

# States for wake word mode
STATE_IDLE = "IDLE"
STATE_LISTENING = "LISTENING"


def main():
    cfg = load_config()

    sample_rate = cfg.get("vad", {}).get("sample_rate", 16000)
    silence_ms = cfg.get("vad", {}).get("silence_threshold_ms", 1500)
    input_device = cfg.get("audio", {}).get("input_device", None)

    # Wake word config
    ww_cfg = cfg.get("wakeword", {})
    wakeword_enabled = ww_cfg.get("enabled", False)
    wakeword_model = ww_cfg.get("model", "hey_jarvis")

    print("═══════════════════════════════════════════")
    print("  voice-in 🎙️  — OpenClaw Voice Frontend")
    print("═══════════════════════════════════════════")
    print(f"  Sample rate:       {sample_rate} Hz")
    print(f"  Silence threshold: {silence_ms} ms")
    print(f"  Wake word:         {'ON (' + wakeword_model + ')' if wakeword_enabled else 'OFF'}")
    input_device = resolve_audio_device(input_device, "input")
    print()

    # Init components
    print("Loading VAD model...")
    vad = SileroVAD(sample_rate=sample_rate)

    wakeword_detector = None
    if wakeword_enabled:
        print(f"Loading wake word model ({wakeword_model})...")
        wakeword_detector = WakeWordDetector(model_name=wakeword_model, sample_rate=sample_rate)

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
    chunk_duration_ms = 32
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    silence_chunks = int(silence_ms / chunk_duration_ms)

    # State
    is_speaking = False
    silence_count = 0
    audio_buffer = []
    state = STATE_IDLE if wakeword_enabled else STATE_LISTENING

    print()
    if wakeword_enabled:
        print(f"🎤 Waiting for wake word... (Ctrl+C to stop)")
    else:
        print("🎤 Listening... (Ctrl+C to stop)")
    print()

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_speaking, silence_count, audio_buffer, state

        if status:
            print(f"  ⚠ Audio: {status}", file=sys.stderr)

        # Ignore audio when muted (voice-out is playing)
        if publisher.is_muted:
            return

        audio = indata[:, 0].copy()  # mono

        # ── IDLE state: only listen for wake word ──
        if state == STATE_IDLE and wakeword_detector is not None:
            if wakeword_detector.detect(audio):
                print("  🔔 Wake word detected!")
                state = STATE_LISTENING
                is_speaking = False
                silence_count = 0
                audio_buffer = []
                vad.reset()
            return

        # ── LISTENING state: VAD + ASR ──
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
                print("  ✋ Silence detected — transcribing...")
                full_audio = np.concatenate(audio_buffer)

                def do_transcribe(audio_data):
                    nonlocal state
                    try:
                        text = asr.transcribe(audio_data, sample_rate)
                        if text:
                            print(f"  📝 \"{text}\"")
                            publisher.publish(text)
                        else:
                            print("  (empty transcription)")
                    except Exception as e:
                        print(f"  ❌ ASR error: {e}", file=sys.stderr)
                    finally:
                        # Return to IDLE if wake word is enabled
                        if wakeword_enabled:
                            state = STATE_IDLE
                            print("  💤 Back to IDLE — waiting for wake word...")

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
            threading.Event().wait()
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        publisher.disconnect()


if __name__ == "__main__":
    main()
