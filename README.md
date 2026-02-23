# 🎙️ openclaw-voice-mqtt

**Voice frontend for OpenClaw — talk to your AI assistant via MQTT**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MQTT](https://img.shields.io/badge/protocol-MQTT-purple.svg)](https://mqtt.org/)

*Turn any device with a mic and speaker into a voice interface for [OpenClaw](https://github.com/openclaw/openclaw) — PC, Raspberry Pi, or anything that runs Python.*

</div>

---

## 🏗️ Architecture

```
┌───────────────────────────────┐              ┌──────────────────────────┐
│  Client (PC / Raspberry Pi)   │    MQTT      │  Server (VPS)            │
│                               │              │                          │
│  ┌─────────┐                  │              │                          │
│  │voice-in │                  │              │                          │
│  │         │                  │              │                          │
│  │ Mic     │                  │  {prefix}/in │                          │
│  │  → VAD  │                  ├─────text────►│  OpenClaw                │
│  │  → ASR  │                  │              │  (MQTT channel plugin)   │
│  │  → Wake │                  │              │                          │
│  └─────────┘                  │              │                          │
│                               │              │                          │
│  ┌─────────┐                  │              │                          │
│  │voice-out│  {prefix}/voice/ │              │                          │
│  │         │◄────text─────────┤◄─────────────┤                          │
│  │ TTS     │       out        │              │                          │
│  │  → Spkr │                  │              │                          │
│  └─────────┘                  │              │                          │
│                               │              │                          │
│  ┌──── anti-loop ────┐        │              │                          │
│  │ voice-out mutes   │        │              │                          │
│  │ voice-in during   │  mute  │              │                          │
│  │ TTS playback      ├───────►│              │                          │
│  └───────────────────┘        │              │                          │
└───────────────────────────────┘              └──────────────────────────┘
```

### Components

| Component | Role |
|-----------|------|
| **voice-in** | Captures mic audio → VAD (Silero) → optional wake word → ASR (Whisper API) → publishes text to MQTT |
| **voice-out** | Subscribes to MQTT reply topic → TTS (OpenAI) → plays audio on speaker |
| **asr-worker** | *(future)* Local ASR server (whisper.cpp) for offline/low-latency transcription |

---

## 📋 Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or higher (tested on 3.10, 3.11, 3.12, 3.13) |
| **OS** | Linux (Ubuntu/Debian/Raspbian), macOS — Windows untested |
| **MQTT broker** | [Mosquitto](https://mosquitto.org/) or any MQTT 3.1.1+ broker |
| **OpenClaw** | With [MQTT channel plugin](https://github.com/openclaw/openclaw) configured |
| **OpenAI API key** | For Whisper (ASR) and TTS |
| **Audio hardware** | Microphone (for voice-in) and speaker (for voice-out) |
| **System libs** | `portaudio19-dev` on Debian/Ubuntu (`sudo apt install portaudio19-dev`) |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/francklefevre/openclaw-voice-mqtt.git
cd openclaw-voice-mqtt
```

### 2. Install dependencies

```bash
./setup.sh
```

This creates a `.venv` virtual environment and installs all dependencies.

> ⚠️ `openwakeword` is installed with `--no-deps` because it depends on `tflite-runtime`, which does not support Python 3.12+. `onnxruntime` is used as the inference backend instead.

### 3. Download wake word models

```bash
.venv/bin/python -c "from openwakeword import utils; utils.download_models()"
```

This downloads the pre-trained ONNX models (including `hey_jarvis`, `alexa`, etc.) to `~/.local/share/openwakeword/`.

### 4. Configure

```bash
cp config.example.yaml config.yaml
nano config.yaml   # Edit with your settings
```

---

## ⚙️ Configuration

All settings live in `config.yaml`. Here's a complete reference:

```yaml
# ── MQTT Broker ─────────────────────────────────────────────────
mqtt:
  broker: "mqtt://your-broker.example.com"   # MQTT broker URL (mqtt:// or mqtts://)
  topic_prefix: "123456/openclaw"            # Prefix for all MQTT topics
                                              # Must match your OpenClaw MQTT plugin config

# ── ASR (Speech-to-Text) ───────────────────────────────────────
asr:
  provider: "openai"               # "openai" (local whisper.cpp planned)
  openai_api_key: "${OPENAI_API_KEY}"  # Reads from environment variable
  model: "whisper-1"               # OpenAI Whisper model
  language: "fr"                   # ISO language code (fr, en, de, es...)

# ── TTS (Text-to-Speech) ───────────────────────────────────────
tts:
  provider: "openai"               # "openai" (elevenlabs, piper planned)
  openai_api_key: "${OPENAI_API_KEY}"
  model: "tts-1"                   # "tts-1" (fast) or "tts-1-hd" (quality)
  voice: "nova"                    # alloy, echo, fable, onyx, nova, shimmer

# ── VAD (Voice Activity Detection) ─────────────────────────────
vad:
  silence_threshold_ms: 1500       # Milliseconds of silence before triggering ASR
                                    # Lower = faster response, higher = fewer false cuts
  sample_rate: 16000               # Audio sample rate in Hz (16000 recommended)

# ── Audio Devices ───────────────────────────────────────────────
audio:
  input_device: null               # Mic device index (null = system default)
  output_device: null              # Speaker device index (null = system default)
                                    # Run `python -m sounddevice` to list devices

# ── Wake Word Detection ────────────────────────────────────────
wakeword:
  enabled: true                    # true = require wake word before listening
                                    # false = always listening (push-to-talk style)
  model: "hey_jarvis"             # Model name: hey_jarvis, alexa, hey_mycroft, etc.
                                    # See: https://github.com/dscripka/openWakeWord
```

> 💡 **Tip:** Use `${ENV_VAR}` syntax for sensitive values. The config loader expands environment variables automatically.

---

## ▶️ Quick Start

### Step 1 — Start voice-out (the speaker)

```bash
.venv/bin/python voice-out/main.py
```

This connects to MQTT and waits for text replies from OpenClaw to speak out loud.

### Step 2 — Start voice-in (the microphone)

In a second terminal:

```bash
.venv/bin/python voice-in/main.py
```

This starts listening for your voice (or waiting for the wake word if enabled).

> 💡 **Tip:** Or activate the venv first (`source .venv/bin/activate`) and use `python` directly.

### Step 3 — Talk!

- If wake word is enabled: say **"Hey Jarvis"**, then speak your question
- If wake word is disabled: just start speaking

Your speech is transcribed and sent to OpenClaw via MQTT. The response is spoken back through your speaker.

---

## 🗣️ Wake Word Detection

Wake word detection uses [OpenWakeWord](https://github.com/dscripka/openWakeWord) to activate voice-in only when a trigger phrase is spoken.

### Available Models

The pre-trained models are **English wake words**:

| Model | Trigger Phrase |
|-------|---------------|
| `hey_jarvis` | "Hey Jarvis" |
| `alexa` | "Alexa" |
| `hey_mycroft` | "Hey Mycroft" |
| `timer` | Timer/alarm sounds |

> 🇬🇧 **Note:** Wake word models are trained on English pronunciation. Even if your assistant speaks French, you'll need to say the wake word in English (e.g., "Hey Jarvis" with English pronunciation). Custom models in other languages can be trained — see the [OpenWakeWord docs](https://github.com/dscripka/openWakeWord).

### Disabling Wake Word

If you don't want wake word detection (always-listening mode), set:

```yaml
wakeword:
  enabled: false
```

In this mode, voice-in listens continuously and transcribes whenever it detects speech.

---

## 🔇 Anti-Loop Mechanism

When voice-out plays audio through the speaker, voice-in would pick it up through the microphone, creating an infinite feedback loop. The anti-loop system prevents this:

1. **Before playback:** voice-out publishes `{"muted": true}` to `{prefix}/voice/mute`
2. **voice-in receives the mute message** and stops processing audio
3. **After playback:** voice-out publishes `{"muted": false}`
4. **voice-in resumes** listening

This happens automatically — no configuration needed.

```
voice-out                          voice-in
   │                                  │
   │  publish mute=true               │
   ├─────────────────────────────────►│ stops listening
   │                                  │
   │  🔊 plays TTS audio             │
   │                                  │
   │  publish mute=false              │
   ├─────────────────────────────────►│ resumes listening
   │                                  │
```

---

## 📨 MQTT Message Format

### voice-in → OpenClaw (`{prefix}/in`)

```json
{
  "text": "Quelle heure est-il ?",
  "replyTopic": "{prefix}/voice/out",
  "sender": "voice-frontend"
}
```

### OpenClaw → voice-out (`{prefix}/voice/out`)

```json
{
  "text": "Il est 14 heures 30.",
  "messageId": "mqtt-1234567890-abc123"
}
```

### Anti-loop mute (`{prefix}/voice/mute`)

```json
{ "muted": true }
```

---

## 🔧 Troubleshooting

### `tflite-runtime` installation fails

**Symptom:** `pip install openwakeword` fails on Python 3.12+

**Solution:** Re-run `./setup.sh`, which installs `openwakeword` with `--no-deps` and uses `onnxruntime` instead.

### No microphone detected

**Symptom:** `sounddevice.PortAudioError: No input device`

**Solution:**
```bash
# Install PortAudio
sudo apt install portaudio19-dev

# List available devices
python -m sounddevice

# Set a specific device in config.yaml
audio:
  input_device: 2   # Use the index from the list above
```

### MQTT connection refused

**Symptom:** `ConnectionRefusedError`

**Solution:** Check that your MQTT broker is running and the URL in `config.yaml` is correct:
```bash
# Test connection
mosquitto_pub -h your-broker.example.com -t test -m "hello"
```

### Wake word not detecting

**Symptom:** Says "Waiting for wake word..." but never triggers

**Possible causes:**
- Speak the wake word clearly in **English** ("Hey Jarvis")
- Check that models are downloaded: `ls ~/.local/share/openwakeword/`
- Try lowering background noise
- Temporarily set `wakeword.enabled: false` to test without it

### Empty transcriptions

**Symptom:** Speech is detected but transcription is empty

**Possible causes:**
- Check your `OPENAI_API_KEY` is set and valid
- Increase `silence_threshold_ms` if speech is being cut off
- Ensure audio input quality is adequate

---

## 🗺️ Roadmap

- [x] voice-in with OpenAI Whisper API
- [x] voice-out with OpenAI TTS
- [x] Anti-loop mute mechanism
- [x] Wake word detection (OpenWakeWord)
- [ ] Local ASR with whisper.cpp (asr-worker)
- [ ] Local TTS with Piper
- [ ] ElevenLabs TTS support
- [ ] Raspberry Pi optimizations
- [ ] Duplex conversation (interrupt while speaking)
- [ ] Web-based audio configuration UI
- [ ] Multi-language wake word models

---

## 📄 License

[MIT](LICENSE) — do whatever you want with it.

---

<div align="center">

Made with ❤️ for the [OpenClaw](https://github.com/openclaw/openclaw) community

</div>
]]>