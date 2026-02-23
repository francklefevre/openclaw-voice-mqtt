#!/usr/bin/env python3
"""
voice: Unified voice frontend for OpenClaw via MQTT.
Handles mic capture, VAD, ASR, MQTT, TTS and speaker playback in one process.
Supports duplex conversation (interrupt by speaking).
"""

import io
import json
import os
import sys
import threading
import wave
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
    """Resolve audio device: try configured, fall back to PulseAudio, then default."""
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


# ── VAD (Silero ONNX) ──────────────────────────────────────────────────────

class SileroVAD:
    """Voice Activity Detection using Silero VAD (ONNX Runtime)."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        import onnxruntime as ort

        self.sample_rate = sample_rate
        self.threshold = threshold
        self._context_size = 64 if sample_rate == 16000 else 32

        model_path = Path(__file__).parent.parent / "voice-in" / "models" / "silero_vad.onnx"
        if not model_path.exists():
            model_path = Path(__file__).parent / "models" / "silero_vad.onnx"
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.reset()

    def is_speech(self, audio_chunk: np.ndarray) -> tuple:
        """Check if an audio chunk contains speech. Returns (is_speech, confidence)."""
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        # Prepend context
        chunk_with_ctx = np.concatenate([self._context, audio_chunk]).astype(np.float32)
        self._context = audio_chunk[-self._context_size:].copy()

        input_data = chunk_with_ctx[np.newaxis, :]
        ort_inputs = {
            "input": input_data,
            "state": self._state,
            "sr": np.array(self.sample_rate, dtype=np.int64),
        }
        ort_out = self.session.run(None, ort_inputs)
        out_val = ort_out[0]
        self._state = ort_out[1]

        confidence = float(out_val.item()) if hasattr(out_val, "item") else float(out_val)
        return confidence > self.threshold, confidence

    def reset(self):
        """Reset VAD state between utterances."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self._context_size, dtype=np.float32)


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


# ── TTS (OpenAI) ───────────────────────────────────────────────────────────

class OpenAITTS:
    """Text-to-Speech using OpenAI API with streaming support."""

    def __init__(self, api_key: str, model: str = "tts-1", voice: str = "nova"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    def synthesize_stream(self, text: str):
        """Stream TTS audio as PCM chunks."""
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4800):
                if chunk:
                    yield chunk


# ── Shared State ────────────────────────────────────────────────────────────

class VoiceState:
    """Shared state between input and output — no MQTT needed for sync."""

    def __init__(self):
        self.is_speaking = False          # True while TTS is playing
        self.interrupted = threading.Event()
        self.interrupt_count = 0          # consecutive speech chunks during playback
        self._lock = threading.Lock()

    def start_speaking(self):
        with self._lock:
            self.interrupted.clear()
            self.interrupt_count = 0
            self.is_speaking = True
        print(f"  🔊 Speaking...")

    def stop_speaking(self):
        with self._lock:
            self.is_speaking = False
        print(f"  🔇 Silent")

    def interrupt(self):
        """Called by mic when speech detected during playback."""
        with self._lock:
            if self.is_speaking:
                self.interrupted.set()
                print(f"  🛑 Interrupted!")
                return True
        return False


# ── Audio Player ────────────────────────────────────────────────────────────

class AudioPlayer:
    """Play PCM audio with interrupt support."""

    def __init__(self, output_device, state: VoiceState):
        self.output_device = output_device
        self.sample_rate = 24000
        self.state = state
        self._lock = threading.Lock()

    def play_stream(self, pcm_chunks):
        """Stream-receive then play. Interruptible."""
        chunks = []
        total_bytes = 0

        for chunk in pcm_chunks:
            if self.state.interrupted.is_set():
                print(f"  ⏹️  Stream interrupted during download")
                return total_bytes
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes == len(chunk):
                print(f"  ⚡ First chunk received...")

        if not chunks or self.state.interrupted.is_set():
            return total_bytes

        print(f"  ▶️  Playing ({total_bytes} bytes)")
        with self._lock:
            all_audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32767.0
            sd.play(all_audio, samplerate=self.sample_rate, device=self.output_device)
            while sd.get_stream().active:
                if self.state.interrupted.is_set():
                    sd.stop()
                    print(f"  ⏹️  Playback stopped")
                    break
                sd.sleep(50)

        return total_bytes


# ── MQTT ────────────────────────────────────────────────────────────────────

class MQTTBridge:
    """MQTT pub/sub for OpenClaw communication."""

    def __init__(self, broker: str, topic_prefix: str, state: VoiceState,
                 tts: OpenAITTS, player: AudioPlayer):
        self.topic_in = f"{topic_prefix}/in"
        self.reply_topic = f"{topic_prefix}/voice/out"
        self.state = state
        self.tts = tts
        self.player = player

        broker_clean = broker.replace("mqtt://", "").replace("mqtts://", "")
        host_parts = broker_clean.split(":")
        host = host_parts[0]
        port = int(host_parts[1]) if len(host_parts) > 1 else 1883

        self.client = mqtt.Client(
            client_id=f"voice-{os.getpid()}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.connect(host, port, keepalive=60)
        self.client.loop_start()

        print(f"  MQTT connected to {host}:{port}")
        print(f"  Publish to:  {self.topic_in}")
        print(f"  Listen on:   {self.reply_topic}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"  🟢 MQTT connected (rc={rc})")
        client.subscribe(self.reply_topic)

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            print(f"  🔴 MQTT disconnected (rc={rc}) — reconnecting...")
        else:
            print(f"  ⚪ MQTT disconnected cleanly")

    def _on_message(self, client, userdata, msg):
        """Handle reply from OpenClaw — synthesize and play."""
        try:
            raw = msg.payload.decode("utf-8")
            try:
                parsed = json.loads(raw)
                text = parsed.get("text", "").strip()
            except json.JSONDecodeError:
                text = raw.strip()

            if not text:
                return

            print(f"  📩 Reply: \"{text}\"")

            def do_speak(t):
                try:
                    self.state.start_speaking()
                    print(f"  🎵 Streaming TTS...")
                    pcm_stream = self.tts.synthesize_stream(t)
                    total = self.player.play_stream(pcm_stream)
                    was_interrupted = self.state.interrupted.is_set()
                    self.state.stop_speaking()
                    if was_interrupted:
                        print(f"  ⏹️  Interrupted after {total} bytes")
                    else:
                        print(f"  ✅ Done ({total} bytes)")
                except Exception as e:
                    self.state.stop_speaking()
                    print(f"  ❌ TTS error: {e}", file=sys.stderr)

            threading.Thread(target=do_speak, args=(text,), daemon=True).start()

        except Exception as e:
            print(f"  ❌ Message error: {e}", file=sys.stderr)

    def publish(self, text: str):
        """Send transcribed text to OpenClaw."""
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
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        prediction = self.model.predict(audio_int16)
        for key in prediction:
            if prediction[key] > self.threshold:
                self.model.reset()
                return True
        return False

    def reset(self):
        self.model.reset()


# ── Main ────────────────────────────────────────────────────────────────────

STATE_IDLE = "IDLE"
STATE_LISTENING = "LISTENING"


def main():
    cfg = load_config()

    vad_cfg = cfg.get("vad", {})
    sample_rate = vad_cfg.get("sample_rate", 16000)
    silence_ms = vad_cfg.get("silence_threshold_ms", 1500)
    vad_threshold = vad_cfg.get("threshold", 0.3)
    input_device = cfg.get("audio", {}).get("input_device", None)
    output_device = cfg.get("audio", {}).get("output_device", None)

    ww_cfg = cfg.get("wakeword", {})
    wakeword_enabled = ww_cfg.get("enabled", False)
    wakeword_model = ww_cfg.get("model", "hey_jarvis")

    print("═══════════════════════════════════════════")
    print("  🎙️🔊 OpenClaw Voice — Unified Frontend")
    print("  Mode: duplex (interruptible)")
    print("═══════════════════════════════════════════")
    print(f"  Sample rate:       {sample_rate} Hz")
    print(f"  Silence threshold: {silence_ms} ms")
    print(f"  VAD threshold:     {vad_threshold}")
    # Interrupt needs N consecutive speech chunks (~N*32ms) to trigger
    interrupt_threshold = vad_cfg.get("interrupt_chunks", 15)  # ~480ms of sustained speech

    print(f"  Wake word:         {'ON (' + wakeword_model + ')' if wakeword_enabled else 'OFF'}")
    print(f"  Interrupt after:   {interrupt_threshold} chunks (~{interrupt_threshold * 32}ms of speech)")
    input_device = resolve_audio_device(input_device, "input")
    output_device = resolve_audio_device(output_device, "output")
    print()

    # Shared state — the magic of single-process
    state = VoiceState()

    # Init components
    print("Loading VAD model...")
    vad = SileroVAD(sample_rate=sample_rate, threshold=vad_threshold)

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

    print("Initializing TTS...")
    tts_cfg = cfg.get("tts", {})
    tts = OpenAITTS(
        api_key=tts_cfg.get("openai_api_key", ""),
        model=tts_cfg.get("model", "tts-1"),
        voice=tts_cfg.get("voice", "nova"),
    )

    player = AudioPlayer(output_device=output_device, state=state)

    print("Connecting to MQTT...")
    mqtt_cfg = cfg.get("mqtt", {})
    bridge = MQTTBridge(
        broker=mqtt_cfg.get("broker", "mqtt://localhost"),
        topic_prefix=mqtt_cfg.get("topic_prefix", "openclaw"),
        state=state,
        tts=tts,
        player=player,
    )

    # VAD parameters
    chunk_duration_ms = 32
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    silence_chunks = int(silence_ms / chunk_duration_ms)

    # State
    is_speaking = False
    silence_count = 0
    audio_buffer = []
    listen_state = STATE_IDLE if wakeword_enabled else STATE_LISTENING

    print()
    if wakeword_enabled:
        print(f"🎤 Waiting for wake word... (Ctrl+C to stop)")
    else:
        print("🎤 Listening... (Ctrl+C to stop)")
    print("💡 Speak while I'm talking to interrupt me!")
    print()

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_speaking, silence_count, audio_buffer, listen_state

        if status:
            print(f"  ⚠ Audio: {status}", file=sys.stderr)

        audio = indata[:, 0].copy()

        # ── Duplex: interrupt if speaking and sustained speech detected ──
        if state.is_speaking:
            if len(audio) >= chunk_samples:
                speech_detected, vad_conf = vad.is_speech(audio[:chunk_samples])
                if speech_detected:
                    state.interrupt_count += 1
                    # Buffer audio from the very first detection
                    audio_buffer.append(audio)
                    # Interrupt playback after sustained speech
                    if state.interrupt_count >= interrupt_threshold:
                        state.interrupt()
                        state.interrupt_count = 0
                        # Transition to LISTENING with buffer intact
                        is_speaking = True
                        silence_count = 0
                        # Don't return — let it fall through to LISTENING
                        # next callback will continue recording
                        return
                else:
                    # Allow small gaps — decay instead of hard reset
                    if state.interrupt_count > 0:
                        audio_buffer.append(audio)  # keep recording during gaps
                        state.interrupt_count = max(0, state.interrupt_count - 1)
                        if state.interrupt_count == 0:
                            audio_buffer.clear()
            return

        # ── IDLE: wake word detection ──
        if listen_state == STATE_IDLE and wakeword_detector is not None:
            if wakeword_detector.detect(audio):
                print("  🔔 Wake word detected!")
                listen_state = STATE_LISTENING
                is_speaking = False
                silence_count = 0
                audio_buffer = []
                vad.reset()
            return

        # ── LISTENING: VAD + ASR ──
        if len(audio) >= chunk_samples:
            speech_detected, vad_conf = vad.is_speech(audio[:chunk_samples])
        else:
            speech_detected, vad_conf = False, 0.0

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
                print("  ✋ Silence — transcribing...")
                full_audio = np.concatenate(audio_buffer)

                def do_transcribe(audio_data):
                    nonlocal listen_state
                    try:
                        text = asr.transcribe(audio_data, sample_rate)
                        if text:
                            print(f"  📝 \"{text}\"")
                            bridge.publish(text)
                        else:
                            print("  (empty transcription)")
                    except Exception as e:
                        print(f"  ❌ ASR error: {e}", file=sys.stderr)
                    finally:
                        if wakeword_enabled:
                            listen_state = STATE_IDLE
                            print("  💤 Waiting for wake word...")

                threading.Thread(target=do_transcribe, args=(full_audio,), daemon=True).start()

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
        bridge.disconnect()


if __name__ == "__main__":
    main()
