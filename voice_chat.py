"""
Real-time Voice Chat Component using FastRTC
=============================================

This script implements a real-time voice chat system for a screenplay-writing assistant.
It uses FastRTC's ReplyOnPause feature to detect when the user has stopped speaking,
then processes their speech through STT -> AI Agent -> TTS pipeline.

Key Components:
- Speech-to-Text (STT): Transcribes user's voice input
- AI Agent: Processes the transcription and generates screenplay-related responses
- Text-to-Speech (TTS): Converts AI responses back to audio
- Gradio UI: Provides web interface for the voice chat

Author: Imaginarium Flamethrower Team
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

try:
    from fastrtc import Stream, AudioDeviceConfig, ReplyOnPause
    from fastrtc.types import AudioData
except ImportError:
    print("Warning: fastrtc not installed. Install with: pip install fastrtc")
    Stream = None
    AudioDeviceConfig = None
    ReplyOnPause = None
    AudioData = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class VoiceChatConfig:
    """Configuration for the voice chat system"""

    # Voice Activity Detection (VAD) settings
    silence_duration: float = 0.5  # Seconds of silence before considering speech ended
    min_speech_duration: float = 0.3  # Minimum speech duration to process

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1

    # AI Agent settings
    screenplay_context: str = "You are a helpful screenplay writing assistant."
    max_response_length: int = 500


# ============================================================================
# STT (Speech-to-Text) - PLUGIN POINT
# ============================================================================

class SpeechToText:
    """
    Speech-to-Text service interface.

    INTEGRATION POINTS:
    -------------------
    Replace this class with your preferred STT provider:

    1. OpenAI Whisper API:
       - Use openai.audio.transcriptions.create()
       - Best for accuracy, supports multiple languages
       - Example: https://platform.openai.com/docs/guides/speech-to-text

    2. Google Cloud Speech-to-Text:
       - Use google.cloud.speech
       - Good for streaming transcription

    3. AssemblyAI:
       - Use assemblyai package
       - Real-time transcription support

    4. Local Whisper:
       - Use whisper package (openai-whisper)
       - Run locally, no API costs
       - Example:
         import whisper
         model = whisper.load_model("base")
         result = model.transcribe(audio_data)
    """

    def __init__(self):
        """Initialize STT service"""
        print("[STT] Initializing Speech-to-Text service...")
        # TODO: Initialize your STT provider here
        # Example: self.client = openai.OpenAI(api_key="...")
        # Example: self.model = whisper.load_model("base")
        pass

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (PCM format, 16kHz, mono)

        Returns:
            Transcribed text string

        IMPLEMENTATION EXAMPLE (Whisper API):
        ------------------------------------
        from openai import OpenAI
        client = OpenAI()

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Convert raw PCM to WAV format
            wav_data = convert_to_wav(audio_data)
            f.write(wav_data)
            temp_path = f.name

        # Transcribe
        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        os.unlink(temp_path)
        return transcription.text
        """

        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate API call

        # Return mock transcription for testing
        mock_text = "User said: [Placeholder - integrate STT here]"
        print(f"[STT] Transcribed: {mock_text}")
        return mock_text


# ============================================================================
# AI Agent - PLUGIN POINT
# ============================================================================

class ScreenplayAgent:
    """
    AI Agent for screenplay writing assistance.

    INTEGRATION POINTS:
    -------------------
    Replace this class with your AI model integration:

    1. OpenAI GPT-4:
       - Use openai.chat.completions.create()
       - Example:
         response = client.chat.completions.create(
             model="gpt-4",
             messages=[{"role": "user", "content": text}]
         )

    2. Anthropic Claude:
       - Use anthropic.Anthropic().messages.create()
       - Great for creative writing tasks

    3. Local LLM (Ollama, LMStudio):
       - Use requests to local API endpoint
       - Run models like Llama, Mistral locally

    4. Your Screenplay Framework:
       - Integrate with your existing screenplay generation models
       - Add character development, plot structure, dialogue tools
       - Could integrate with beat sheets, character arcs, etc.
    """

    def __init__(self, config: VoiceChatConfig):
        """Initialize AI agent"""
        self.config = config
        self.conversation_history = []
        print("[AI Agent] Initializing Screenplay Assistant...")

        # TODO: Initialize your AI model here
        # Example: self.client = anthropic.Anthropic(api_key="...")
        # Example: self.ollama_url = "http://localhost:11434/api/chat"

    async def process(self, user_input: str) -> str:
        """
        Process user input and generate screenplay-related response.

        Args:
            user_input: Transcribed text from user

        Returns:
            AI-generated response text

        IMPLEMENTATION EXAMPLE (OpenAI):
        --------------------------------
        from openai import OpenAI
        client = OpenAI()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Generate response
        response = client.chat.completions.create(
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

        SCREENPLAY-SPECIFIC FEATURES:
        -----------------------------
        You can extend this agent with:
        - Character development tools
        - Plot structure analysis
        - Dialogue enhancement
        - Scene description generation
        - Beat sheet integration
        - Three-act structure guidance
        """

        # Placeholder implementation
        await asyncio.sleep(0.2)  # Simulate processing

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Generate mock response
        response = f"I understand you mentioned: '{user_input}'. How can I help with your screenplay?"

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        print(f"[AI Agent] Generated response: {response}")
        return response


# ============================================================================
# TTS (Text-to-Speech) - PLUGIN POINT
# ============================================================================

class TextToSpeech:
    """
    Text-to-Speech service interface.

    INTEGRATION POINTS:
    -------------------
    Replace this class with your preferred TTS provider:

    1. OpenAI TTS:
       - Use openai.audio.speech.create()
       - Natural-sounding voices (alloy, echo, fable, onyx, nova, shimmer)
       - Example:
         response = client.audio.speech.create(
             model="tts-1",
             voice="alloy",
             input=text
         )

    2. ElevenLabs:
       - Use elevenlabs package
       - High-quality, customizable voices
       - Good for character voice differentiation

    3. Google Cloud TTS:
       - Use google.cloud.texttospeech
       - Multiple languages and voices

    4. Edge TTS (Free):
       - Use edge-tts package
       - Free Microsoft TTS service
       - Example:
         import edge_tts
         communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
         await communicate.save("output.mp3")
    """

    def __init__(self):
        """Initialize TTS service"""
        print("[TTS] Initializing Text-to-Speech service...")
        # TODO: Initialize your TTS provider here
        # Example: self.client = openai.OpenAI(api_key="...")
        pass

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio.

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes (PCM format, 16kHz, mono)

        IMPLEMENTATION EXAMPLE (OpenAI TTS):
        -----------------------------------
        from openai import OpenAI
        import io

        client = OpenAI()

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # or "echo", "fable", "onyx", "nova", "shimmer"
            input=text,
            response_format="pcm"  # or "mp3", "opus", "aac", "flac"
        )

        # Get audio bytes
        audio_bytes = response.content

        # Convert to 16kHz mono PCM if needed
        # (depends on TTS output format)

        return audio_bytes

        SCREENPLAY-SPECIFIC FEATURES:
        -----------------------------
        For screenplay applications, you might want:
        - Different voices for different characters
        - Emotion/tone control for dialogue
        - Voice direction hints (e.g., "angry", "whispered")
        """

        # Placeholder implementation
        await asyncio.sleep(0.2)  # Simulate API call

        # Generate silence as placeholder (1 second of 16kHz mono PCM)
        # In production, this would be actual synthesized speech
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)

        # Create silent audio (zeros)
        audio_data = bytes(num_samples * 2)  # 2 bytes per sample (16-bit)

        print(f"[TTS] Synthesized: {text}")
        return audio_data


# ============================================================================
# FastRTC Voice Chat Handler
# ============================================================================

class VoiceChatHandler:
    """
    Main handler for real-time voice chat using FastRTC.

    This class orchestrates the STT -> AI Agent -> TTS pipeline
    using FastRTC's ReplyOnPause feature for voice activity detection.
    """

    def __init__(self, config: Optional[VoiceChatConfig] = None):
        """Initialize voice chat handler"""
        self.config = config or VoiceChatConfig()

        # Initialize components
        self.stt = SpeechToText()
        self.agent = ScreenplayAgent(self.config)
        self.tts = TextToSpeech()

        # State
        self._process_lock = asyncio.Lock()
        self.stream: Optional[Stream] = None

        print("[Voice Chat] Initialized successfully")

    async def on_audio_chunk(self, audio_data: Any) -> None:
        """
        Called when user speaks (pause detected).

        This is the main callback for ReplyOnPause.
        It receives audio data when the user stops speaking.

        Args:
            audio_data: Audio data from FastRTC (user's speech)
        """
        if self._process_lock.locked():
            print("[Voice Chat] Already processing, skipping...")
            return

        async with self._process_lock:
            print("\n[Voice Chat] User finished speaking, processing...")

            try:
                audio_bytes = self._extract_audio_bytes(audio_data)
            except ValueError as exc:
                print(f"[Voice Chat] Could not extract audio: {exc}")
                return

            if not audio_bytes or len(audio_bytes) < 1000:
                print("[Voice Chat] Audio too short, skipping...")
                return

            # Step 2: Speech-to-Text
            print("[Voice Chat] Step 1/3: Transcribing speech...")
            try:
                transcription = await self.stt.transcribe(audio_bytes)
            except Exception as exc:  # pragma: no cover - depends on provider
                print(f"[Voice Chat] STT failed: {exc}")
                return

            if not transcription.strip():
                print("[Voice Chat] No speech detected")
                return

            # Step 3: AI Agent processing
            print("[Voice Chat] Step 2/3: Processing with AI agent...")
            try:
                response_text = await self.agent.process(transcription)
            except Exception as exc:  # pragma: no cover - depends on provider
                print(f"[Voice Chat] Agent error: {exc}")
                return

            # Step 4: Text-to-Speech
            print("[Voice Chat] Step 3/3: Generating speech response...")
            try:
                response_audio = await self.tts.synthesize(response_text)
            except Exception as exc:  # pragma: no cover - depends on provider
                print(f"[Voice Chat] TTS failed: {exc}")
                return

            if not response_audio:
                print("[Voice Chat] No audio generated from TTS")
                return

            # Step 5: Send audio back through FastRTC stream
            if self.stream:
                try:
                    await self._send_audio(response_audio)
                    print("[Voice Chat] Response sent to user")
                except Exception as exc:  # pragma: no cover - depends on FastRTC
                    print(f"[Voice Chat] Failed to send audio: {exc}")

    def _extract_audio_bytes(self, audio_data: Any) -> bytes:
        """Extract raw audio bytes from supported FastRTC payloads."""
        if audio_data is None:
            raise ValueError("No audio data provided")

        if isinstance(audio_data, bytes):
            return audio_data

        if isinstance(audio_data, (bytearray, memoryview)):
            return bytes(audio_data)

        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            # Common pattern: (sample_rate, samples)
            _, payload = audio_data
            return self._audio_array_to_bytes(payload)

        if AudioData is not None and isinstance(audio_data, AudioData):
            if hasattr(audio_data, "pcm"):
                pcm = audio_data.pcm
                return bytes(pcm) if not isinstance(pcm, bytes) else pcm

            if hasattr(audio_data, "data"):
                data = audio_data.data
                if isinstance(data, (bytes, bytearray, memoryview)):
                    return bytes(data)
                return self._audio_array_to_bytes(data)

            if hasattr(audio_data, "tobytes"):
                return audio_data.tobytes()

        if hasattr(audio_data, "tobytes"):
            return audio_data.tobytes()

        if isinstance(audio_data, (list, tuple)):
            return self._audio_array_to_bytes(audio_data)

        raise ValueError(f"Unsupported audio data type: {type(audio_data)!r}")

    def _audio_array_to_bytes(self, payload: Any) -> bytes:
        """Convert array-like audio payloads to 16-bit PCM bytes."""
        array = np.asarray(payload)

        if array.size == 0:
            raise ValueError("Audio input contained no samples")

        if array.ndim > 1:
            array = np.mean(array, axis=1)

        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, -1.0, 1.0)
            array = (array * np.iinfo(np.int16).max).astype(np.int16)
        else:
            array = array.astype(np.int16, copy=False)

        return array.tobytes()

    async def _send_audio(self, audio_bytes: bytes) -> None:
        """
        Send audio response back to user through FastRTC stream.

        Args:
            audio_bytes: Audio data to send

        NOTE: This is a placeholder. The actual implementation depends on
        FastRTC's Stream API for sending audio. Check documentation for
        methods like stream.send(), stream.write(), etc.
        """
        # Placeholder - adjust based on FastRTC API
        if hasattr(self.stream, 'send'):
            await self.stream.send(audio_bytes)
        elif hasattr(self.stream, 'write'):
            await self.stream.write(audio_bytes)
        else:
            print("[Voice Chat] Warning: Don't know how to send audio via stream")

    async def start(self) -> None:
        """
        Start the voice chat stream using FastRTC.

        This sets up the FastRTC Stream with ReplyOnPause for voice detection.
        """
        if Stream is None:
            raise ImportError("FastRTC not installed. Run: pip install fastrtc")

        print("[Voice Chat] Starting FastRTC stream...")

        # Configure audio device
        # Adjust these settings based on your audio hardware and requirements
        audio_config = AudioDeviceConfig(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )

        # Create ReplyOnPause handler
        # This automatically detects when user stops speaking
        reply_handler = ReplyOnPause(
            on_audio=self.on_audio_chunk,
            silence_duration=self.config.silence_duration,
            min_speech_duration=self.config.min_speech_duration,
        )

        # Create and start stream
        self.stream = Stream(
            audio_config=audio_config,
            on_audio=reply_handler,
        )

        print("[Voice Chat] Stream started! Listening for speech...")
        print("[Voice Chat] Speak into your microphone. I'll respond when you pause.")

        await self.stream.run()

    async def stop(self) -> None:
        """Stop the voice chat stream"""
        if self.stream:
            print("[Voice Chat] Stopping stream...")
            await self.stream.stop()
            self.stream = None


# ============================================================================
# Gradio UI Integration
# ============================================================================

def create_gradio_interface():
    """
    Create Gradio UI for the voice chat system.

    GRADIO INTEGRATION:
    -------------------
    This function sets up a web interface using Gradio for the voice chat.

    To run with Gradio:
    1. Install: pip install gradio
    2. Run this function: demo = create_gradio_interface()
    3. Launch: demo.launch()

    The UI provides:
    - Start/Stop controls
    - Real-time transcription display
    - Conversation history
    - Configuration options

    CUSTOMIZATION:
    --------------
    You can enhance the UI with:
    - Screenplay-specific controls (character selection, scene context)
    - Visual feedback for voice activity
    - Export conversation to screenplay format
    - Save/load conversation sessions
    """

    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        return None

    # Global state
    chat_handler = None
    conversation_log = []

    async def start_chat():
        """Start voice chat session"""
        nonlocal chat_handler

        if chat_handler is not None:
            return "Voice chat already running!", conversation_log

        try:
            chat_handler = VoiceChatHandler()

            # Start in background
            asyncio.create_task(chat_handler.start())

            message = "Voice chat started! Speak into your microphone."
            conversation_log.append(("System", message))

            return message, conversation_log

        except Exception as e:
            error_msg = f"Error starting chat: {e}"
            conversation_log.append(("System", error_msg))
            return error_msg, conversation_log

    async def stop_chat():
        """Stop voice chat session"""
        nonlocal chat_handler

        if chat_handler is None:
            return "No active voice chat session.", conversation_log

        try:
            await chat_handler.stop()
            chat_handler = None

            message = "Voice chat stopped."
            conversation_log.append(("System", message))

            return message, conversation_log

        except Exception as e:
            error_msg = f"Error stopping chat: {e}"
            return error_msg, conversation_log

    def clear_history():
        """Clear conversation history"""
        nonlocal conversation_log
        conversation_log = []
        return "History cleared.", []

    # Create Gradio interface
    with gr.Blocks(title="Screenplay Voice Assistant") as demo:
        gr.Markdown("# 🎬 Screenplay Writing Voice Assistant")
        gr.Markdown(
            "Real-time voice chat for screenplay development. "
            "Speak naturally and get AI-powered screenplay assistance."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Controls")

                start_btn = gr.Button("🎤 Start Voice Chat", variant="primary")
                stop_btn = gr.Button("⏹️ Stop Voice Chat", variant="stop")
                clear_btn = gr.Button("🗑️ Clear History")

                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to start",
                    interactive=False
                )

                gr.Markdown("### Configuration")
                gr.Markdown(
                    "Configure STT, TTS, and AI agent in `voice_chat.py`\n\n"
                    "See comments in code for integration instructions."
                )

            with gr.Column(scale=2):
                gr.Markdown("### Conversation")

                conversation_box = gr.Chatbot(
                    label="Conversation History",
                    height=500
                )

                gr.Markdown(
                    "**Note:** This is a demo UI. The actual voice interaction "
                    "happens through your microphone/speakers, not this interface. "
                    "The conversation history will appear here once the backend "
                    "is fully integrated."
                )

        # Wire up event handlers
        start_btn.click(
            fn=start_chat,
            inputs=[],
            outputs=[status_text, conversation_box]
        )

        stop_btn.click(
            fn=stop_chat,
            inputs=[],
            outputs=[status_text, conversation_box]
        )

        clear_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[status_text, conversation_box]
        )

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """
    Main entry point for voice chat.

    Run this for command-line voice chat (no Gradio UI).
    """
    print("=" * 60)
    print("Screenplay Writing Voice Assistant")
    print("=" * 60)
    print()

    # Create and start voice chat
    handler = VoiceChatHandler()

    try:
        await handler.start()
    except KeyboardInterrupt:
        print("\n[Voice Chat] Interrupted by user")
    finally:
        await handler.stop()


def main_gradio():
    """
    Main entry point for Gradio UI.

    Run this to launch the web interface.
    """
    demo = create_gradio_interface()

    if demo is None:
        print("Error: Could not create Gradio interface")
        print("Install gradio: pip install gradio")
        return

    # Launch Gradio
    print("Launching Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True to create public link
    )


if __name__ == "__main__":
    import sys

    print("Screenplay Voice Assistant")
    print()
    print("Choose mode:")
    print("1. Command-line voice chat (default)")
    print("2. Gradio web interface")
    print()

    mode = input("Enter mode (1 or 2): ").strip()

    if mode == "2":
        main_gradio()
    else:
        # Run command-line mode
        asyncio.run(main())
