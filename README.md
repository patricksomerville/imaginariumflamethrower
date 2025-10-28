# Imaginarium Flamethrower 🎬

Real-time Voice Chat Component for Screenplay Writing Assistant

This project provides a production-ready voice chat system built with FastRTC, designed for creating an interactive screenplay writing assistant. It uses voice activity detection (ReplyOnPause) to create natural conversational interactions.

## Features

- **Real-time Voice Activity Detection**: Uses FastRTC's ReplyOnPause to automatically detect when the user stops speaking
- **Modular Architecture**: Easy-to-integrate STT, TTS, and AI agent components
- **Gradio Web Interface**: Beautiful web UI for managing voice chat sessions
- **Screenplay-Focused**: Designed specifically for screenplay writing assistance
- **Plugin-Ready**: Clear integration points for your preferred STT, TTS, and LLM providers

## Architecture

The system follows a simple pipeline:

```
User Speech → [VAD] → [STT] → [AI Agent] → [TTS] → Audio Response
             FastRTC   ↓         ↓            ↓
                       Plugin   Plugin      Plugin
                       Points   Points      Points
```

### Components

1. **VoiceChatHandler**: Orchestrates the entire pipeline
2. **SpeechToText**: Converts audio to text (plugin point)
3. **ScreenplayAgent**: Processes text and generates responses (plugin point)
4. **TextToSpeech**: Converts text back to audio (plugin point)
5. **Gradio UI**: Web interface for managing sessions

## Installation

### Quick Start (Minimal)

```bash
# Install core dependencies
pip install fastrtc gradio

# Run the demo
python voice_chat.py
```

### Recommended Setup (OpenAI Integration)

```bash
# Install with OpenAI for STT, TTS, and LLM
pip install fastrtc gradio openai pydub numpy scipy

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the application
python voice_chat.py
```

### Full Local Setup (No API Costs)

```bash
# Install local models
pip install fastrtc gradio
pip install openai-whisper torch transformers
pip install pyttsx3 pydub numpy scipy

# Install Ollama for local LLM
# Download from: https://ollama.ai

# Pull a model (e.g., llama2)
ollama pull llama2

# Run the application
python voice_chat.py
```

### From Requirements File

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Usage

### Command-Line Mode

Run the voice chat directly from the command line:

```bash
python voice_chat.py
# Select option 1 when prompted
```

This mode:
- Listens to your microphone continuously
- Detects when you stop speaking (pause detection)
- Processes your speech through STT → AI → TTS
- Plays the response through your speakers

### Gradio Web Interface Mode

Launch the web interface:

```bash
python voice_chat.py
# Select option 2 when prompted
```

Or directly:

```python
from voice_chat import main_gradio
main_gradio()
```

This will start a web server at `http://localhost:7860` with:
- Start/Stop voice chat controls
- Real-time status updates
- Conversation history display
- Configuration options

### Integration in Your Own Code

```python
import asyncio
from voice_chat import VoiceChatHandler, VoiceChatConfig

async def main():
    # Create custom configuration
    config = VoiceChatConfig(
        silence_duration=0.5,  # Seconds of silence to detect speech end
        min_speech_duration=0.3,  # Minimum speech duration to process
        screenplay_context="You are a screenplay consultant specializing in dramatic structure.",
    )

    # Create handler
    handler = VoiceChatHandler(config)

    # Start voice chat
    await handler.start()

# Run
asyncio.run(main())
```

## Configuration

### Voice Activity Detection

Adjust pause detection in `VoiceChatConfig`:

```python
config = VoiceChatConfig(
    silence_duration=0.5,     # How long to wait after user stops talking
    min_speech_duration=0.3,  # Minimum speech length to process
)
```

### Audio Settings

Configure audio quality:

```python
config = VoiceChatConfig(
    sample_rate=16000,  # 16kHz (standard for speech)
    channels=1,         # Mono audio
)
```

### Screenplay Context

Customize the AI assistant's behavior:

```python
config = VoiceChatConfig(
    screenplay_context="""You are an expert screenplay consultant.
    Help with: plot structure, character development, dialogue, and formatting.
    Reference Save the Cat, Story, and other screenplay principles.""",
    max_response_length=500,  # Token limit for responses
)
```

## Integration Points

The code has clear integration points for your preferred services:

### 1. Speech-to-Text (STT)

Edit the `SpeechToText.transcribe()` method in `voice_chat.py`:

**Option A: OpenAI Whisper API**
```python
from openai import OpenAI

async def transcribe(self, audio_data: bytes) -> str:
    client = OpenAI()

    # Save to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        wav_data = convert_to_wav(audio_data)
        f.write(wav_data)
        f.seek(0)

        # Transcribe
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    return transcript.text
```

**Option B: Local Whisper**
```python
import whisper

def __init__(self):
    self.model = whisper.load_model("base")

async def transcribe(self, audio_data: bytes) -> str:
    # Save to temp file and transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        wav_data = convert_to_wav(audio_data)
        f.write(wav_data)

        result = self.model.transcribe(f.name)

    return result["text"]
```

**Option C: AssemblyAI**
```python
import assemblyai as aai

def __init__(self):
    aai.settings.api_key = "your-api-key"
    self.transcriber = aai.Transcriber()

async def transcribe(self, audio_data: bytes) -> str:
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        wav_data = convert_to_wav(audio_data)
        f.write(wav_data)

        transcript = self.transcriber.transcribe(f.name)

    return transcript.text
```

### 2. AI Agent / LLM

Edit the `ScreenplayAgent.process()` method:

**Option A: OpenAI GPT-4**
```python
from openai import OpenAI

def __init__(self, config):
    self.client = OpenAI()
    self.config = config
    self.conversation_history = []

async def process(self, user_input: str) -> str:
    self.conversation_history.append({
        "role": "user",
        "content": user_input
    })

    response = self.client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": self.config.screenplay_context},
            *self.conversation_history
        ],
        max_tokens=self.config.max_response_length
    )

    assistant_message = response.choices[0].message.content

    self.conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message
```

**Option B: Anthropic Claude**
```python
import anthropic

def __init__(self, config):
    self.client = anthropic.Anthropic()
    self.config = config
    self.conversation_history = []

async def process(self, user_input: str) -> str:
    self.conversation_history.append({
        "role": "user",
        "content": user_input
    })

    message = self.client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=self.config.max_response_length,
        system=self.config.screenplay_context,
        messages=self.conversation_history
    )

    assistant_message = message.content[0].text

    self.conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message
```

**Option C: Local LLM (Ollama)**
```python
import requests

def __init__(self, config):
    self.config = config
    self.ollama_url = "http://localhost:11434/api/chat"
    self.conversation_history = []

async def process(self, user_input: str) -> str:
    self.conversation_history.append({
        "role": "user",
        "content": user_input
    })

    response = requests.post(
        self.ollama_url,
        json={
            "model": "llama2",
            "messages": [
                {"role": "system", "content": self.config.screenplay_context},
                *self.conversation_history
            ],
            "stream": False
        }
    )

    assistant_message = response.json()["message"]["content"]

    self.conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message
```

### 3. Text-to-Speech (TTS)

Edit the `TextToSpeech.synthesize()` method:

**Option A: OpenAI TTS**
```python
from openai import OpenAI

def __init__(self):
    self.client = OpenAI()

async def synthesize(self, text: str) -> bytes:
    response = self.client.audio.speech.create(
        model="tts-1",
        voice="alloy",  # or "echo", "fable", "onyx", "nova", "shimmer"
        input=text,
        response_format="pcm"
    )

    return response.content
```

**Option B: ElevenLabs**
```python
from elevenlabs import generate, set_api_key

def __init__(self):
    set_api_key("your-api-key")

async def synthesize(self, text: str) -> bytes:
    audio = generate(
        text=text,
        voice="Bella",  # or other voice IDs
        model="eleven_monolingual_v1"
    )

    return audio
```

**Option C: Edge TTS (Free)**
```python
import edge_tts

async def synthesize(self, text: str) -> bytes:
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")

    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    return audio_data
```

**Option D: pyttsx3 (Offline)**
```python
import pyttsx3
import io

def __init__(self):
    self.engine = pyttsx3.init()

async def synthesize(self, text: str) -> bytes:
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    self.engine.save_to_file(text, temp_path)
    self.engine.runAndWait()

    # Read back as bytes
    with open(temp_path, "rb") as f:
        audio_data = f.read()

    os.unlink(temp_path)
    return audio_data
```

## Screenplay-Specific Features

This system is designed specifically for screenplay writing. Here are some ideas for extending it:

### Character Voice Differentiation

Use different TTS voices for different characters:

```python
class ScreenplayTTS:
    def __init__(self):
        self.character_voices = {
            "protagonist": "alloy",
            "antagonist": "onyx",
            "narrator": "nova",
        }

    async def synthesize(self, text: str, character: str = "narrator") -> bytes:
        voice = self.character_voices.get(character, "alloy")
        # Use appropriate TTS with selected voice
        ...
```

### Beat Sheet Integration

Integrate with screenplay structure:

```python
class ScreenplayAgent:
    def __init__(self, config):
        self.beat_sheet = {
            "act_1": ["opening_image", "theme_stated", "catalyst", "debate"],
            "act_2": ["break_into_two", "b_story", "midpoint", "all_is_lost"],
            "act_3": ["break_into_three", "finale", "final_image"],
        }
        self.current_beat = "opening_image"

    async def process(self, user_input: str) -> str:
        # Consider current beat in screenplay structure
        context = f"Currently working on: {self.current_beat}"
        # Generate contextually appropriate advice
        ...
```

### Dialogue Enhancement

Analyze and improve dialogue:

```python
async def enhance_dialogue(self, dialogue: str, character: str) -> str:
    prompt = f"""
    Enhance this dialogue for {character}:
    "{dialogue}"

    Consider:
    - Character voice and personality
    - Subtext and conflict
    - Rhythm and pacing
    - Show don't tell
    """
    return await self.agent.process(prompt)
```

### Scene Description Generation

Generate scene descriptions from verbal descriptions:

```python
async def generate_scene_description(self, verbal_description: str) -> str:
    prompt = f"""
    Convert this verbal description into a screenplay scene description:
    "{verbal_description}"

    Format: Standard screenplay scene heading and action.
    Style: Visual, present tense, active voice.
    """
    return await self.agent.process(prompt)
```

## Troubleshooting

### No Audio Input/Output

Check your audio device configuration:

```python
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set specific device in AudioDeviceConfig
config = AudioDeviceConfig(
    input_device=1,   # Device ID from list above
    output_device=2,  # Device ID from list above
    sample_rate=16000,
    channels=1,
)
```

### FastRTC Import Errors

Ensure FastRTC is installed correctly:

```bash
pip install --upgrade fastrtc
```

If issues persist, check FastRTC documentation for platform-specific requirements.

### STT/TTS API Errors

Verify API keys are set:

```bash
export OPENAI_API_KEY="your-key"
export ELEVENLABS_API_KEY="your-key"
export ASSEMBLYAI_API_KEY="your-key"
```

### Latency Issues

Reduce latency by:
1. Using faster models (e.g., `tts-1` instead of `tts-1-hd`)
2. Shortening `silence_duration` in config
3. Running models locally instead of API calls
4. Using streaming STT/TTS if available

## File Structure

```
imaginariumflamethrower/
├── voice_chat.py          # Main voice chat implementation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Development

### Running Tests

```bash
# TODO: Add test suite
pytest tests/
```

### Code Style

```bash
# Format code
black voice_chat.py

# Lint
flake8 voice_chat.py

# Type check
mypy voice_chat.py
```

## Roadmap

- [ ] Add streaming STT for lower latency
- [ ] Implement character voice management
- [ ] Add screenplay export functionality
- [ ] Create beat sheet tracking system
- [ ] Add conversation save/load
- [ ] Implement multi-user collaboration
- [ ] Add screenplay formatting tools
- [ ] Create character profile system
- [ ] Add plot structure analysis

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - feel free to use in your own projects!

## Resources

- [FastRTC Documentation](https://github.com/fastrtc/fastrtc)
- [Gradio Documentation](https://gradio.app/docs)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Screenplay Format Guide](https://www.scriptreaderpro.com/screenplay-format/)
- [Save the Cat Beat Sheet](https://savethecat.com/beat-sheet)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review FastRTC documentation
3. Open an issue in this repository

## Acknowledgments

Built with:
- FastRTC for real-time communication
- Gradio for the web interface
- OpenAI for STT/TTS/LLM capabilities (optional)

Created for the Imaginarium Flamethrower project - making screenplay writing more accessible through voice interaction.
