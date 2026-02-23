# 🎙️ openclaw-voice-mqtt

**Voice frontend for OpenClaw — talk to your AI assistant via MQTT**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MQTT](https://img.shields.io/badge/protocol-MQTT-purple.svg)](https://mqtt.org/)

*Turn any device with a mic and speaker into a voice interface for [OpenClaw](https://github.com/openclaw/openclaw) — PC, Raspberry Pi, or anything that runs Python.*

---

## 🏗️ Architecture

```
┌───────────────────────────────┐              ┌──────────────────────────┐
│  Client (PC / Raspberry Pi)   │    MQTT      │  Server (VPS)            │
│                               │              │                          │
│  ┌─────────┐                  │              │                          │
│  │voice-in │  {prefix}/in     │              │                          │
│  │         ├─────text────────►│─────────────►│  OpenClaw                │
│  │ Mic     │                  │              │  (MQTT channel plugin)   │
│  │  → VAD  │                  │              │                          │
│  │  → ASR  │                  │              │                          │
│  └─────────┘                  │              │                          │
│                               │              │                          │
│  ┌─────────┐  {prefix}/voice/ │              │                          │
│  │voice-out│◄────text─────────┤◄─────────────┤                          │
│  │         │       out        │              │                          │
│  │ TTS     │                  │              │                          │
│  │  → Spkr │                  │              │                          │
│  └─────────┘                  │              │                          │
│                               │              │                          │
│  ┌──── duplex ───────┐        │              │                          │
│  │ voice-in stays    │        │              │                          │
│  │ active during TTS │ speak/ │              │                          │
│  │ detects speech →  │ inter- │              │                          │
│  │ sends interrupt   │ rupt   │              │                          │
│  └───────────────────┘        │              │                          │
└───────────────────────────────┘              └──────────────────────────┘
```

### Components

| Component | Role |
|-----------|------|
| **voice-in** | Captures mic audio → VAD (Silero ONNX) → optional wake word → ASR (Whisper API) → publishes text to MQTT |
| **voice-out** | Subscribes to MQTT reply topic → TTS (OpenAI streaming) → interruptible audio playback |

---

## ✨ Features

- **Duplex conversation** — speak while the assistant is talking to interrupt it instantly
- **Streaming TTS** — audio starts playing as soon as the first chunks arrive from OpenAI
- **Silero VAD via ONNX Runtime** — lightweight voice activity detection, works on Raspberry Pi (no PyTorch needed)
- **Wake word detection** — optional activation via OpenWakeWord (English models: "Hey Jarvis", "Alexa", etc.)
- **PulseAudio support** — auto-detects PulseAudio/PipeWire for Bluetooth headset compatibility
- **Auto-reconnect** — MQTT connection is monitored and restored automatically
- **Runs on minimal hardware** — tested on Raspberry Pi 3 (1 GB RAM) with Bluetooth headset

---

## 📋 Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10+ (tested on 3.10–3.13) |
| **OS** | Linux (Ubuntu/Debian/Raspbian), macOS — Windows untested |
| **MQTT broker** | [Mosquitto](https://mosquitto.org/) or any MQTT 3.1.1+ broker |
| **OpenClaw** | With [MQTT channel plugin](https://github.com/openclaw/openclaw) configured |
| **OpenAI API key** | For Whisper (ASR) and TTS |
| **Audio hardware** | Microphone + speaker (or Bluetooth headset) |
| **System libs** | `portaudio19-dev` on Debian/Ubuntu (`sudo apt install portaudio19-dev`) |

---

## 🚀 Installation

### 1. Clone

```bash
git clone https://github.com/francklefevre/openclaw-voice-mqtt.git
cd openclaw-voice-mqtt
```

### 2. Install dependencies

```bash
./setup.sh
```

Creates a `.venv` and installs everything. `openwakeword` is installed with `--no-deps` (uses `onnxruntime` instead of `tflite-runtime` for Python 3.12+ compatibility).

### 3. Download wake word models (optional)

```bash
.venv/bin/python -c "from openwakeword import utils; utils.download_models()"
```

### 4. Configure

```bash
cp config.example.yaml config.yaml
nano config.yaml
```

---

## ⚙️ Configuration

All settings in `config.yaml`:

```yaml
# ── MQTT Broker ─────────────────────────────────────────────────
mqtt:
  broker: "mqtt://your-broker.example.com"
  topic_prefix: "123456/openclaw"           # Must match OpenClaw MQTT plugin

# ── ASR (Speech-to-Text) ───────────────────────────────────────
asr:
  provider: "openai"
  openai_api_key: "${OPENAI_API_KEY}"
  model: "whisper-1"
  language: "fr"                            # ISO language code

# ── TTS (Text-to-Speech) ───────────────────────────────────────
tts:
  provider: "openai"
  openai_api_key: "${OPENAI_API_KEY}"
  model: "tts-1"                            # tts-1 (fast) or tts-1-hd (quality)
  voice: "nova"                             # alloy, echo, fable, onyx, nova, shimmer

# ── VAD (Voice Activity Detection) ─────────────────────────────
vad:
  silence_threshold_ms: 1500                # ms of silence before triggering ASR
  sample_rate: 16000
  threshold: 0.3                            # VAD confidence threshold (0.0–1.0)

# ── Audio Devices ───────────────────────────────────────────────
audio:
  input_device: null                        # null = auto-detect (prefers PulseAudio)
  output_device: null                       # Run `python -m sounddevice` to list

# ── Wake Word Detection ────────────────────────────────────────
wakeword:
  enabled: false                            # true = require wake word before listening
  model: "hey_jarvis"                       # hey_jarvis, alexa, hey_mycroft, etc.
```

> 💡 Use `${ENV_VAR}` syntax for sensitive values — the config loader expands environment variables.

---

## ▶️ Quick Start

```bash
# Terminal 1 — speaker
.venv/bin/python voice-out/main.py

# Terminal 2 — microphone
.venv/bin/python voice-in/main.py
```

Then just talk! Your speech is transcribed and sent to OpenClaw via MQTT. The response is spoken back through your speaker.

If the assistant is speaking and you start talking, it will **stop immediately** and listen to you (duplex mode).

---

## 🔄 Duplex Conversation

The voice frontend supports **full duplex** — you can interrupt the assistant while it's speaking.

### How it works

1. **voice-out** publishes `{"speaking": true}` on `{prefix}/voice/speaking` when TTS starts
2. **voice-in** stays active and monitors the microphone via VAD
3. When voice-in detects speech during playback, it publishes `{"interrupt": true}` on `{prefix}/voice/interrupt`
4. **voice-out** receives the interrupt and stops playback immediately (`sd.stop()`)
5. **voice-out** publishes `{"speaking": false}` — voice-in resumes normal processing

```
voice-in                          voice-out
   │                                  │
   │       speaking=true              │
   │◄─────────────────────────────────┤ starts TTS playback
   │  (still listening via VAD)       │ 🔊
   │                                  │
   │  🗣️ user speaks                 │
   │  interrupt=true                  │
   ├─────────────────────────────────►│ ⏹️ stops playback
   │                                  │
   │       speaking=false             │
   │◄─────────────────────────────────┤
   │                                  │
   │  (processes speech normally)     │
```

> 💡 **Bluetooth headsets** don't create feedback loops (the mic doesn't pick up speaker output), so no echo cancellation is needed.

---

## 📨 MQTT Topics

| Topic | Direction | Payload |
|-------|-----------|---------|
| `{prefix}/in` | voice-in → OpenClaw | `{"text": "...", "replyTopic": "{prefix}/voice/out", "sender": "voice-frontend"}` |
| `{prefix}/voice/out` | OpenClaw → voice-out | `{"text": "...", "messageId": "mqtt-..."}` |
| `{prefix}/voice/speaking` | voice-out → voice-in | `{"speaking": true/false}` |
| `{prefix}/voice/interrupt` | voice-in → voice-out | `{"interrupt": true}` |

---

## 🍓 Raspberry Pi Setup

Tested on **Raspberry Pi 3** (1 GB RAM, Raspbian 64-bit) with a Bluetooth headset.

### Key points

- **No PyTorch** — Silero VAD runs via ONNX Runtime (the `silero_vad.onnx` model is embedded in `voice-in/models/`)
- **PulseAudio required** for Bluetooth audio — the `pulse` device is auto-detected
- **`openwakeword`** installed with `--no-deps` + `onnxruntime` (no `tflite-runtime`)

### Bluetooth headset setup

```bash
# Install PulseAudio + Bluetooth
sudo apt install pulseaudio pulseaudio-module-bluetooth bluez

# Pair your headset
bluetoothctl
> scan on
> pair XX:XX:XX:XX:XX:XX
> connect XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX

# Verify audio devices
python -m sounddevice
# Look for "pulse" device — it should route to your headset
```

### Auto-reconnect tip

Add `module-switch-on-connect` to PulseAudio to auto-route audio when the headset reconnects:

```bash
# In /etc/pulse/default.pa or ~/.config/pulse/default.pa
load-module module-switch-on-connect
```

---

## 🔧 Troubleshooting

### `tflite-runtime` installation fails
Re-run `./setup.sh` — it installs `openwakeword` with `--no-deps` and uses `onnxruntime` instead.

### No microphone detected
```bash
sudo apt install portaudio19-dev
python -m sounddevice            # List devices
# Set input_device in config.yaml if needed
```

### MQTT connection refused
Check broker URL and test with: `mosquitto_pub -h broker.example.com -t test -m "hello"`

### Wake word not detecting
- Wake words are **English-only** — pronounce "Hey Jarvis" in English
- Check models are downloaded: `ls ~/.local/share/openwakeword/`
- Try disabling wake word (`wakeword.enabled: false`)

### Whisper transcribes garbage on silence
Whisper sometimes hallucinates on silence (e.g., "Sous-titres réalisés par..."). This is a known Whisper behavior — increasing the VAD threshold helps filter out near-silence segments.

### No sound on Bluetooth (RPi)
- Ensure PulseAudio is running: `pulseaudio --check && echo OK`
- Check headset is connected: `pactl list sinks short`
- Restart voice-out after reconnecting the headset
- Add `module-switch-on-connect` (see RPi section above)

---

## 🗺️ Roadmap

- [x] Voice-in with OpenAI Whisper API
- [x] Voice-out with OpenAI streaming TTS
- [x] Silero VAD via ONNX Runtime (RPi-compatible)
- [x] Wake word detection (OpenWakeWord)
- [x] Duplex conversation (interrupt while speaking)
- [x] Raspberry Pi 3 + Bluetooth headset support
- [x] PulseAudio auto-detection
- [x] MQTT auto-reconnect
- [ ] Whisper hallucination filtering
- [ ] Local ASR with whisper.cpp
- [ ] Local TTS with Piper
- [ ] ElevenLabs TTS support
- [ ] systemd services for auto-start
- [ ] Multi-language wake word models

---

## 📄 License

[MIT](LICENSE)

---

<div align="center">

Made with ❤️ for the [OpenClaw](https://github.com/openclaw/openclaw) community

</div>
