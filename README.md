# openclaw-voice-mqtt 🎙️⚡

Voice frontend for [OpenClaw](https://github.com/openclaw/openclaw) via MQTT.

Talk to your OpenClaw assistant with your voice — from a PC, Raspberry Pi, or any device with a mic and speaker.

## Architecture

```
┌──────────────────────┐           ┌─────────────────────────┐
│  PC / Raspberry Pi   │   MQTT    │  Server (VPS)           │
│                      │           │                         │
│  voice-in            │           │                         │
│  Mic → VAD → ASR ────┼──text────►│  OpenClaw               │
│                      │           │  (MQTT channel plugin)  │
│  voice-out           │           │                         │
│  Speaker ← TTS ◄─────┼──text────┤                         │
└──────────────────────┘           └─────────────────────────┘
```

### Components

- **voice-in**: Captures audio from microphone, detects speech/silence (VAD), sends to ASR (Whisper API), publishes transcribed text to MQTT
- **voice-out**: Subscribes to MQTT reply topic, sends text to TTS API, plays audio on local speaker
- **asr-worker** *(future)*: Local ASR server (whisper.cpp) for offline/low-latency transcription

## Quick Start

### Prerequisites

- Python 3.10+
- An MQTT broker (e.g., Mosquitto)
- OpenClaw with [MQTT channel plugin](https://github.com/openclaw/openclaw) configured
- OpenAI API key (for Whisper ASR and TTS)

### Install

```bash
./setup.sh
```

This creates a `.venv` virtual environment and installs all dependencies.

> **Note:** `openwakeword` is installed with `--no-deps` because it pulls
> `tflite-runtime` which doesn't support Python 3.12+ on Linux.
> `onnxruntime` is used instead.

### Configure

Copy and edit the config file:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your MQTT broker, topics, and API keys.

### Run

Terminal 1 — listen for replies and play them:
```bash
.venv/bin/python voice-out/main.py
```

Terminal 2 — capture voice and send to OpenClaw:
```bash
.venv/bin/python voice-in/main.py
```

> Or activate the venv first (`source .venv/bin/activate`) and use `python` directly.

## Configuration

```yaml
mqtt:
  broker: "mqtt://k1tests.k1info.com"
  topic_prefix: "123456/openclaw"

asr:
  provider: "openai"          # openai | local (future)
  openai_api_key: "${OPENAI_API_KEY}"
  model: "whisper-1"
  language: "fr"

tts:
  provider: "openai"          # openai | elevenlabs | piper (future)
  openai_api_key: "${OPENAI_API_KEY}"
  model: "tts-1"
  voice: "nova"

vad:
  silence_threshold_ms: 1500  # silence duration to trigger ASR
  sample_rate: 16000

audio:
  input_device: null           # null = system default
  output_device: null           # null = system default
```

## MQTT Message Format

### voice-in publishes to `{prefix}/in`:
```json
{
  "text": "Quelle heure est-il ?",
  "replyTopic": "{prefix}/voice/out",
  "sender": "voice-frontend"
}
```

### voice-out subscribes to `{prefix}/voice/out`:
```json
{
  "text": "Il est 9h du matin.",
  "messageId": "mqtt-1234567890-abc123"
}
```

## Roadmap

- [ ] Basic voice-in with Whisper API
- [ ] Basic voice-out with OpenAI TTS
- [ ] Local ASR with whisper.cpp (asr-worker)
- [ ] Local TTS with piper
- [ ] Wake word detection
- [ ] Raspberry Pi optimizations
- [ ] Duplex conversation (interrupt while speaking)

## License

MIT
