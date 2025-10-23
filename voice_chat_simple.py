"""
Simple Voice Chat Example - Ready to Run!
==========================================

This is a simplified, working version of the voice chat system that uses
OpenAI for everything (Whisper for STT, GPT-4 for AI, TTS for voice).

Just run: python voice_chat_simple.py
Then open: http://localhost:7860
"""

import os
import asyncio
import tempfile
import io
from pathlib import Path

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables only.")

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("\n" + "="*60)
    print("ERROR: No OpenAI API key found!")
    print("="*60)
    print("\nPlease set your API key using ONE of these methods:")
    print("\n1. Create a .env file with:")
    print("   OPENAI_API_KEY=your-key-here")
    print("\n2. Set environment variable:")
    print("   export OPENAI_API_KEY='your-key-here'  (Mac/Linux)")
    print("   set OPENAI_API_KEY=your-key-here  (Windows)")
    print("\n" + "="*60 + "\n")
    exit(1)

try:
    from openai import OpenAI
    import gradio as gr
except ImportError as e:
    print("\n" + "="*60)
    print("ERROR: Missing required packages!")
    print("="*60)
    print(f"\nMissing: {e.name}")
    print("\nPlease install required packages:")
    print("  pip install openai gradio")
    print("\nOr install everything:")
    print("  pip install fastrtc gradio openai python-dotenv")
    print("\n" + "="*60 + "\n")
    exit(1)

# Initialize OpenAI client
client = OpenAI()

print("\n" + "="*60)
print("Simple Voice Chat - Initializing...")
print("="*60)

# ============================================================================
# Speech-to-Text using OpenAI Whisper
# ============================================================================

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribe audio using OpenAI Whisper API.

    Args:
        audio_file_path: Path to audio file

    Returns:
        Transcribed text
    """
    try:
        print(f"[STT] Transcribing audio from: {audio_file_path}")

        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"  # Optional: specify language
            )

        text = transcript.text
        print(f"[STT] Transcribed: {text}")
        return text

    except Exception as e:
        print(f"[STT] Error: {e}")
        return f"Error transcribing audio: {e}"


# ============================================================================
# AI Agent using GPT-4
# ============================================================================

# Conversation history (maintains context)
conversation_history = []

def process_with_ai(user_text: str) -> str:
    """
    Process user input with GPT-4.

    Args:
        user_text: User's transcribed speech

    Returns:
        AI-generated response
    """
    try:
        print(f"[AI] Processing: {user_text}")

        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_text
        })

        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-turbo" or "gpt-3.5-turbo" for faster/cheaper
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful screenplay writing assistant.
                    Help users develop characters, plot structure, dialogue, and scenes.
                    Be encouraging and creative. Keep responses concise (2-3 sentences)
                    since they will be spoken aloud."""
                },
                *conversation_history
            ],
            max_tokens=150,  # Keep responses short for voice
            temperature=0.7
        )

        # Get assistant's response
        assistant_message = response.choices[0].message.content

        # Add to history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        print(f"[AI] Response: {assistant_message}")
        return assistant_message

    except Exception as e:
        print(f"[AI] Error: {e}")
        return f"Sorry, I encountered an error: {e}"


# ============================================================================
# Text-to-Speech using OpenAI TTS
# ============================================================================

def synthesize_speech(text: str) -> str:
    """
    Convert text to speech using OpenAI TTS.

    Args:
        text: Text to convert to speech

    Returns:
        Path to generated audio file
    """
    try:
        print(f"[TTS] Synthesizing: {text}")

        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality (slower)
            voice="nova",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text,
        )

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(response.content)
        temp_file.close()

        print(f"[TTS] Generated audio: {temp_file.name}")
        return temp_file.name

    except Exception as e:
        print(f"[TTS] Error: {e}")
        return None


# ============================================================================
# Gradio Interface
# ============================================================================

def process_audio(audio_input):
    """
    Main processing pipeline: Audio -> STT -> AI -> TTS -> Audio

    Args:
        audio_input: Audio from Gradio (file path or tuple)

    Returns:
        tuple: (transcription, ai_response, response_audio_path)
    """
    if audio_input is None:
        return "No audio provided", "Please record or upload audio first.", None

    # Handle different audio input formats from Gradio
    if isinstance(audio_input, tuple):
        # Format: (sample_rate, audio_data)
        audio_path = audio_input
    else:
        # Already a file path
        audio_path = audio_input

    print("\n" + "-"*60)
    print("Processing new audio input...")
    print("-"*60)

    try:
        # Step 1: Transcribe audio
        transcription = transcribe_audio(audio_path)

        if not transcription or "Error" in transcription:
            return transcription, "Could not transcribe audio", None

        # Step 2: Process with AI
        ai_response = process_with_ai(transcription)

        # Step 3: Convert response to speech
        response_audio = synthesize_speech(ai_response)

        print("-"*60)
        print("Processing complete!")
        print("-"*60 + "\n")

        return transcription, ai_response, response_audio

    except Exception as e:
        error_msg = f"Error in pipeline: {e}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return "Error", error_msg, None


def reset_conversation():
    """Reset the conversation history"""
    global conversation_history
    conversation_history = []
    return "Conversation reset! Start fresh.", ""


# ============================================================================
# Create Gradio UI
# ============================================================================

def create_interface():
    """Create the Gradio web interface"""

    with gr.Blocks(
        title="Screenplay Voice Assistant",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("# 🎬 Screenplay Voice Assistant")
        gr.Markdown(
            "Talk to your AI screenplay assistant! Record your voice, "
            "get AI-powered feedback, and hear the response."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎤 Your Voice Input")

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload audio"
                )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Process", variant="primary", scale=2)
                    reset_btn = gr.Button("🔄 Reset Conversation", scale=1)

                transcription_box = gr.Textbox(
                    label="📝 What you said (Transcription)",
                    lines=3,
                    interactive=False
                )

            with gr.Column():
                gr.Markdown("### 🤖 AI Response")

                response_box = gr.Textbox(
                    label="💬 AI's Response",
                    lines=5,
                    interactive=False
                )

                response_audio = gr.Audio(
                    label="🔊 Listen to Response",
                    type="filepath",
                    interactive=False
                )

        gr.Markdown("---")

        with gr.Accordion("📖 How to Use", open=False):
            gr.Markdown("""
            1. **Click the microphone icon** above and allow browser access
            2. **Record your question** about screenplay writing
            3. **Click "Process"** and wait a few seconds
            4. **See the transcription** of what you said
            5. **Read the AI's response** and **listen to it** spoken aloud
            6. **Continue the conversation** - the AI remembers context!

            **Tips:**
            - Ask about plot structure, character development, dialogue, scenes, etc.
            - Speak clearly and avoid background noise
            - Keep questions concise for faster processing
            - Use "Reset Conversation" to start a fresh topic

            **Example questions:**
            - "How do I structure Act 2 of my screenplay?"
            - "Help me develop my protagonist's character arc"
            - "What makes good dialogue?"
            - "I'm writing a scene in a coffee shop, what should I include?"
            """)

        with gr.Accordion("⚙️ Configuration", open=False):
            gr.Markdown(f"""
            **Current Settings:**
            - Speech-to-Text: OpenAI Whisper
            - AI Model: GPT-4
            - Text-to-Speech: OpenAI TTS (nova voice)
            - API Key: {"✓ Loaded" if os.getenv("OPENAI_API_KEY") else "✗ Not found"}

            To change these, edit `voice_chat_simple.py`
            """)

        # Wire up the interface
        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[transcription_box, response_box, response_audio]
        )

        reset_btn.click(
            fn=reset_conversation,
            inputs=[],
            outputs=[response_box, transcription_box]
        )

    return demo


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nCreating interface...")
    demo = create_interface()

    print("\n" + "="*60)
    print("🚀 Starting Gradio server...")
    print("="*60)
    print("\n📱 Open in your browser: http://localhost:7860")
    print("\n💡 Tips:")
    print("   - Allow microphone access when prompted")
    print("   - Speak clearly into your microphone")
    print("   - The first response may be slower (initializing)")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("\n" + "="*60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public URL (useful for testing on mobile)
        show_error=True
    )
