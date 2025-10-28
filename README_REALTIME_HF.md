# Real-time Voice Chat with FastRTC + Hugging Face

This is the **TRUE real-time streaming** implementation you requested! Not record-and-respond.

## What This Does

This implements a continuous audio stream where:
1. **FastRTC** captures your voice in real-time
2. **ReplyOnPause** automatically detects when you stop speaking
3. **Hugging Face Whisper** transcribes your speech instantly
4. **Hugging Face Zephyr-7B** generates AI responses
5. **Hugging Face SpeechT5** converts responses to speech
6. Audio streams back to you automatically

## Key Difference from `voice_chat_simple.py`

**voice_chat_simple.py**: Record → Click Submit → Process → Response
**voice_chat_realtime_hf.py**: Continuous stream → Auto-detect pause → Auto-respond

This is like talking to a person, not recording a voice message!

## Installation

### 1. Install Dependencies

```bash
cd ~/imaginariumflamethrower
source venv/bin/activate
pip install transformers torch torchaudio datasets accelerate sentencepiece
```

All dependencies (including fastrtc and gradio) should already be installed from earlier.

### 2. Models Downloaded Automatically

The first time you run this, it will download:
- Whisper-Tiny (150MB) - Speech-to-text
- Zephyr-7B (14GB) - AI conversation model
- SpeechT5 (500MB) - Text-to-speech

**Note:** Zephyr-7B is large! If you have limited disk space or RAM, you can modify the code to use a smaller model like:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (2GB)
- `microsoft/DialoGPT-medium` (1.5GB)

## Running

```bash
cd ~/imaginariumflamethrower
source venv/bin/activate
python voice_chat_realtime_hf.py
```

Then open: **http://localhost:7860**

## How to Use

1. **Click "Start Stream"** - This begins real-time listening
2. **Speak naturally** - Just talk! No buttons to press
3. **Pause when done** - The system detects silence (0.8 seconds)
4. **AI responds automatically** - Voice response plays back
5. **Keep talking** - It's a continuous conversation
6. **Click "Stop Stream"** when finished

## Architecture

### FastRTC Stream Flow

```
[Your Microphone]
       ↓
[FastRTC Stream] ← Continuous audio capture
       ↓
[ReplyOnPause] ← Detects when you stop speaking (VAD)
       ↓
[Audio Chunk] ← Your complete sentence
       ↓
[HuggingFaceSTT] ← Whisper transcription
       ↓
[User Text]
       ↓
[HuggingFaceLLM] ← Zephyr-7B generates response
       ↓
[AI Text Response]
       ↓
[HuggingFaceTTS] ← SpeechT5 synthesis
       ↓
[Audio Response]
       ↓
[FastRTC Stream] ← Sends audio back to you
       ↓
[Your Speakers]
```

### ReplyOnPause Parameters

Configured in `RTCConfig`:

```python
silence_duration: float = 0.8  # Seconds of silence before processing
min_speech_duration: float = 0.3  # Minimum speech duration to process
```

You can adjust these if the system:
- Cuts you off too early → Increase `silence_duration`
- Takes too long to respond → Decrease `silence_duration`
- Triggers on background noise → Increase `min_speech_duration`

## Performance

### CPU vs GPU

**With GPU (CUDA):**
- Models run on GPU
- Fast inference (1-2 seconds per response)
- Recommended for real-time feel

**With CPU:**
- Models run on CPU
- Slower inference (5-10 seconds per response)
- Still works, just not as "real-time"

The code automatically detects and uses GPU if available:

```python
device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

### Memory Requirements

- **Minimum**: 8GB RAM (CPU mode with smaller models)
- **Recommended**: 16GB RAM + 8GB VRAM GPU
- **Storage**: 15GB for default models

### Optimizations You Can Make

1. **Use smaller models** (edit `RTCConfig`):
   ```python
   stt_model: str = "openai/whisper-tiny.en"  # Already smallest
   llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 2GB
   tts_model: str = "microsoft/speecht5_tts"  # Already small
   ```

2. **Quantization** (add to model loading):
   ```python
   load_in_8bit=True  # Halves memory usage
   load_in_4bit=True  # Quarters memory usage
   ```

3. **Batch size** (reduce for lower memory):
   ```python
   batch_size=8  # In STT pipeline
   ```

## Troubleshooting

### Models downloading too slowly?
Set Hugging Face mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Out of memory error?
Use smaller models or quantization (see Optimizations above).

### Audio choppy or laggy?
- Close other apps using microphone
- Reduce `sample_rate` in config to 8000
- Use GPU acceleration

### "CUDA out of memory"?
```python
# Edit RTCConfig
device: str = "cpu"  # Force CPU mode
```

### No audio heard from AI?
- Check speaker/volume
- Check browser audio permissions
- Look for errors in terminal about audio device

### FastRTC import error?
```bash
pip install fastrtc
```

## Customization

### Change AI Personality

Edit the system prompt in `HuggingFaceLLM.process()`:

```python
messages = [
    {"role": "system", "content": "You are a helpful screenplay writing assistant."},
    # Change to whatever you want!
]
```

### Use Different Models

Edit `RTCConfig` class:

```python
stt_model: str = "openai/whisper-base.en"  # Better quality
llm_model: str = "meta-llama/Llama-2-7b-chat-hf"  # If you have access
tts_model: str = "suno/bark"  # Different TTS (larger but better)
```

### Add Screenplay-Specific Features

The code is designed to be extended. Examples:

1. **Character Voice Selection**:
   ```python
   # In HuggingFaceTTS, change speaker_embeddings
   self.speaker_embeddings = load_character_voice(character_name)
   ```

2. **Scene Context**:
   ```python
   # In HuggingFaceLLM, add screenplay context
   messages = [
       {"role": "system", "content": f"You are in Scene {scene_num}..."},
       ...
   ]
   ```

3. **Beat Sheet Integration**:
   ```python
   # Pass beat sheet to LLM for context-aware responses
   ```

## Comparison with Other Files

**voice_chat.py**: Template with placeholders (you fill in STT/TTS/LLM)
**voice_chat_simple.py**: OpenAI APIs, record-and-respond
**voice_chat_realtime_hf.py**: This file - FastRTC + Hugging Face, TRUE real-time streaming

## Next Steps

Once this works, you can:

1. **Integrate with screenplay-storyboard-app** for full screenplay generation
2. **Add character voices** from your existing TTS services
3. **Connect to Manus** for workflow automation
4. **Add beat sheet awareness** for structured story development
5. **Multi-character conversations** with different voices

## License

Part of the Imaginarium Flamethrower project.

---

**Questions?** The code is heavily commented. Read through `voice_chat_realtime_hf.py` for implementation details.

**Issues?** Check the Troubleshooting section above or look at the terminal output for error messages.
