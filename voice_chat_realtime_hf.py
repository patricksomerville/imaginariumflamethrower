"""
Real-time Voice Chat with FastRTC and Hugging Face Models
===========================================================

This implements a TRUE real-time voice chat system using:
- FastRTC: Real-time audio streaming with ReplyOnPause for voice activity detection
- Hugging Face Whisper: Speech-to-text transcription
- Hugging Face LLM: AI conversation (Mistral/Llama)
- Hugging Face TTS: Text-to-speech synthesis

This is NOT record-and-respond. It's continuous streaming audio.
"""

import asyncio
import numpy as np
import torch
import io
import wave
from typing import Optional, List
from dataclasses import dataclass

# Check for required packages
try:
    from fastrtc import Stream, ReplyOnPause
except ImportError:
    print("ERROR: FastRTC not installed!")
    print("Install: pip install fastrtc")
    exit(1)

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio not installed!")
    print("Install: pip install gradio")
    exit(1)

try:
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline
    )
except ImportError:
    print("ERROR: Transformers not installed!")
    print("Install: pip install transformers torch torchaudio")
    exit(1)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RTCConfig:
    """Real-time chat configuration"""

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1

    # Voice Activity Detection
    silence_duration: float = 0.8  # Seconds of silence before processing
    min_speech_duration: float = 0.3  # Minimum speech to process

    # Model settings
    stt_model: str = "openai/whisper-tiny.en"  # Fast STT model
    llm_model: str = "HuggingFaceH4/zephyr-7b-beta"  # Conversational LLM
    tts_model: str = "microsoft/speecht5_tts"  # Fast TTS

    # Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_bettertransformer: bool = True  # Speed optimization


# ============================================================================
# Hugging Face STT (Speech-to-Text)
# ============================================================================

class HuggingFaceSTT:
    """Real-time speech-to-text using Hugging Face Whisper"""

    def __init__(self, config: RTCConfig):
        self.config = config
        print(f"[STT] Loading Whisper model: {config.stt_model}")
        print(f"[STT] Using device: {config.device}")

        # Load Whisper model for speech recognition
        self.processor = AutoProcessor.from_pretrained(config.stt_model)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.stt_model,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(config.device)

        # Use pipeline for easier inference
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            device=config.device,
        )

        print("[STT] Whisper model loaded successfully")

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Run transcription
            result = self.pipe(audio_array, generate_kwargs={"language": "english"})

            text = result["text"].strip()
            print(f"[STT] Transcribed: {text}")
            return text

        except Exception as e:
            print(f"[STT] Error: {e}")
            return ""


# ============================================================================
# Hugging Face LLM (AI Agent)
# ============================================================================

class HuggingFaceLLM:
    """Conversational AI using Hugging Face LLM"""

    def __init__(self, config: RTCConfig):
        self.config = config
        self.conversation_history: List[dict] = []

        print(f"[LLM] Loading model: {config.llm_model}")
        print(f"[LLM] Using device: {config.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if config.device == "cuda" else None
        )

        if config.device == "cpu":
            self.model.to(config.device)

        print("[LLM] Model loaded successfully")

    async def process(self, user_text: str) -> str:
        """Process user input and generate response"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_text
            })

            # Build prompt with system message
            messages = [
                {"role": "system", "content": "You are a helpful screenplay writing assistant. Keep responses concise and creative."},
                *self.conversation_history[-6:]  # Last 6 messages for context
            ]

            # Format prompt
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the assistant's response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            else:
                # Fallback: get text after the prompt
                response = response[len(prompt):].strip()

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            print(f"[LLM] Response: {response[:100]}...")
            return response

        except Exception as e:
            print(f"[LLM] Error: {e}")
            return "I'm sorry, I encountered an error processing that."


# ============================================================================
# Hugging Face TTS (Text-to-Speech)
# ============================================================================

class HuggingFaceTTS:
    """Text-to-speech using Hugging Face models"""

    def __init__(self, config: RTCConfig):
        self.config = config
        print(f"[TTS] Loading model: {config.tts_model}")

        # Use SpeechT5 for TTS
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset

        self.processor = SpeechT5Processor.from_pretrained(config.tts_model)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(config.tts_model)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        self.model.to(config.device)
        self.vocoder.to(config.device)
        self.speaker_embeddings = self.speaker_embeddings.to(config.device)

        print("[TTS] Model loaded successfully")

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech audio bytes"""
        try:
            # Truncate if too long
            if len(text) > 600:
                text = text[:600] + "..."

            # Process text
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    self.speaker_embeddings,
                    vocoder=self.vocoder
                )

            # Convert to bytes
            audio_array = (speech.cpu().numpy() * 32767).astype(np.int16)

            # Create WAV file in memory
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_array.tobytes())

            audio_bytes = buffer.getvalue()
            print(f"[TTS] Generated {len(audio_bytes)} bytes of audio")
            return audio_bytes

        except Exception as e:
            print(f"[TTS] Error: {e}")
            return b''


# ============================================================================
# Real-time Voice Chat Handler
# ============================================================================

class RealtimeVoiceChat:
    """Main handler for real-time voice chat with FastRTC"""

    def __init__(self, config: Optional[RTCConfig] = None):
        self.config = config or RTCConfig()

        print("\n" + "="*60)
        print("INITIALIZING REAL-TIME VOICE CHAT")
        print("="*60)

        # Initialize AI components
        self.stt = HuggingFaceSTT(self.config)
        self.llm = HuggingFaceLLM(self.config)
        self.tts = HuggingFaceTTS(self.config)

        # State
        self.stream: Optional[Stream] = None
        self.is_processing = False
        self.active = False

        print("\n[RTC] ✓ All components initialized")
        print("="*60 + "\n")

    def on_user_speech(self, audio_data: tuple):
        """
        Callback when user finishes speaking (ReplyOnPause triggered).
        This is where the real-time magic happens!

        Args:
            audio_data: Tuple of (sample_rate, numpy_array) from FastRTC
        """
        if self.is_processing:
            print("[RTC] Busy processing, skipping...")
            return None

        try:
            self.is_processing = True
            print("\n[RTC] 🎤 User finished speaking...")

            # Extract audio from FastRTC format
            sample_rate, audio_array = audio_data

            # Convert numpy array to bytes
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()

            if len(audio_bytes) < 4000:  # Too short
                print("[RTC] Audio too short, skipping")
                return None

            # Step 1: Speech-to-Text (synchronous in this context)
            print("[RTC] [1/3] Transcribing speech...")
            # Run async function in sync context for FastRTC
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            user_text = loop.run_until_complete(self.stt.transcribe(audio_bytes))
            loop.close()

            if not user_text.strip():
                print("[RTC] No speech detected")
                return None

            # Step 2: AI Response
            print("[RTC] [2/3] Generating AI response...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ai_response = loop.run_until_complete(self.llm.process(user_text))
            loop.close()

            # Step 3: Text-to-Speech
            print("[RTC] [3/3] Converting to speech...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_audio = loop.run_until_complete(self.tts.synthesize(ai_response))
            loop.close()

            # Step 4: Return audio to FastRTC
            if response_audio:
                # Convert bytes to numpy array for FastRTC
                response_array = np.frombuffer(response_audio[44:], dtype=np.int16).astype(np.float32) / 32767.0  # Skip WAV header
                print("[RTC] ✓ Returning audio response\n")
                return (16000, response_array)  # Return (sample_rate, numpy_array)

        except Exception as e:
            print(f"[RTC] ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.is_processing = False

        return None



# ============================================================================
# Gradio UI with FastRTC
# ============================================================================

def create_gradio_interface() -> gr.Blocks:
    """Create Gradio web interface for the real-time voice chat"""

    # Initialize chat handler
    print("Initializing Real-time Voice Chat Handler...")
    chat_handler = RealtimeVoiceChat()

    with gr.Blocks(title="Real-time Voice Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎙️ Real-time Voice Chat with FastRTC + Hugging Face

        This is a **true real-time** voice chat system using:
        - **FastRTC** for streaming audio with voice activity detection
        - **Hugging Face Whisper** for speech-to-text
        - **Hugging Face Zephyr** for AI responses
        - **Hugging Face SpeechT5** for text-to-speech

        ### How to Use:
        1. Click "Connect" below to start the audio stream
        2. Speak naturally - the system detects when you finish speaking
        3. AI responds automatically with voice
        4. The conversation continues until you close the page

        ### ⚠️ First Run:
        Models will download automatically (~15GB). This may take 10-30 minutes.
        """)

        # Create FastRTC Stream with ReplyOnPause
        handler = ReplyOnPause(
            fn=chat_handler.on_user_speech,
            input_sample_rate=16000,
            output_sample_rate=16000
        )

        webrtc = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio"
        )

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" REAL-TIME VOICE CHAT - FastRTC + Hugging Face")
    print("="*70)
    print("\nMode: Real-time streaming (NOT record-and-respond)")
    print(f"Device: {RTCConfig().device}")
    print(f"Models: Whisper-Tiny + Zephyr-7B + SpeechT5")
    print("\n" + "="*70 + "\n")

    # Create and launch Gradio interface
    demo = create_gradio_interface()

    print("\n🚀 Launching Gradio interface...")
    print("📱 Open: http://localhost:7860")
    print("\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
