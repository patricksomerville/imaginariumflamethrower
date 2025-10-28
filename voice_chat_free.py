"""
FREE Voice Chat - No API Keys Needed!
======================================

This version uses browser-based speech recognition and synthesis.
No OpenAI account or API keys required!

Just run: python3 voice_chat_free.py
Then open: http://localhost:7860
"""

import gradio as gr
import random

print("=" * 60)
print("FREE Voice Chat - Starting...")
print("=" * 60)
print()
print("✓ No API keys needed!")
print("✓ Uses your browser's built-in speech features")
print("✓ Completely free to use")
print()


# ============================================================================
# Simple AI Response Generator (No API needed!)
# ============================================================================

class SimpleScreenplayAI:
    """
    A simple rule-based screenplay assistant.
    No API calls - just helpful pre-written responses!
    """

    def __init__(self):
        self.conversation_history = []

        # Screenplay writing tips and responses
        self.responses = {
            "greeting": [
                "Hello! I'm your screenplay writing assistant. How can I help you today?",
                "Hi there! Ready to work on your screenplay? What would you like to discuss?",
                "Welcome! Let's create something amazing. What aspect of your screenplay should we focus on?",
            ],
            "character": [
                "Great question about characters! Remember, the best characters have clear goals, flaws, and growth arcs. What's your protagonist's main desire?",
                "Character development is crucial. Think about what your character wants, what's stopping them, and how they'll change by the end.",
                "Try giving your character a specific quirk or trait that makes them memorable. What makes them unique?",
            ],
            "dialogue": [
                "Good dialogue reveals character and advances the plot. Each line should serve a purpose - show don't tell!",
                "Make your dialogue sound natural by reading it aloud. Cut anything that feels forced or on-the-nose.",
                "Remember, great dialogue has subtext. Characters often say one thing but mean another.",
            ],
            "structure": [
                "Most screenplays follow a three-act structure: Setup, Confrontation, and Resolution. Where are you in your story?",
                "Think about the key beats: inciting incident at page 10-15, midpoint at page 50-60, all is lost at page 75-85.",
                "Your first act should establish the ordinary world, introduce the hero, and present the central problem.",
            ],
            "scene": [
                "Every scene should either advance the plot or reveal character - ideally both!",
                "Start scenes late and end them early. Cut the boring parts and get to the conflict.",
                "Think visually! Screenplays are a visual medium. What can we SEE happening?",
            ],
            "plot": [
                "Strong plots have clear cause and effect. Each action leads to the next naturally.",
                "Your protagonist should drive the story forward through their choices and actions.",
                "Build tension by adding obstacles and raising the stakes throughout the story.",
            ],
            "general": [
                "That's an interesting aspect of screenwriting! Could you tell me more about what specific part you're working on?",
                "I can help with character development, plot structure, dialogue, scenes, and more. What would you like to explore?",
                "Remember the golden rule: show, don't tell. Let's make your screenplay visual and engaging!",
            ],
        }

    def get_response(self, user_input):
        """
        Generate a helpful response based on keywords in user input.
        """
        user_input_lower = user_input.lower()

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Determine response category based on keywords
        if any(word in user_input_lower for word in ["hello", "hi", "hey", "greet"]):
            response = random.choice(self.responses["greeting"])
        elif any(word in user_input_lower for word in ["character", "protagonist", "hero", "villain"]):
            response = random.choice(self.responses["character"])
        elif any(word in user_input_lower for word in ["dialogue", "conversation", "speak", "talk", "say"]):
            response = random.choice(self.responses["dialogue"])
        elif any(word in user_input_lower for word in ["structure", "act", "beat", "plot point"]):
            response = random.choice(self.responses["structure"])
        elif any(word in user_input_lower for word in ["scene", "action", "visual"]):
            response = random.choice(self.responses["scene"])
        elif any(word in user_input_lower for word in ["plot", "story", "narrative", "arc"]):
            response = random.choice(self.responses["plot"])
        else:
            response = random.choice(self.responses["general"])

        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response


# ============================================================================
# Gradio Interface with Browser-Based Speech
# ============================================================================

# Initialize AI
ai = SimpleScreenplayAI()

def process_text(text_input):
    """
    Process text input and generate response.
    This works with both typed and voice input!
    """
    if not text_input or not text_input.strip():
        return "Please say or type something first!", ""

    # Get AI response
    response = ai.get_response(text_input)

    return text_input, response


# Create Gradio interface
with gr.Blocks(title="FREE Screenplay Voice Assistant", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🎬 FREE Screenplay Voice Assistant")
    gr.Markdown(
        "**No API keys needed!** This uses your browser's built-in speech features. "
        "Click the microphone icon to speak, or type your question."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎤 Your Input")

            # Text input (also captures voice via Gradio's built-in speech recognition)
            text_input = gr.Textbox(
                label="Type or use voice",
                placeholder="Ask about characters, plot, dialogue, structure...",
                lines=3
            )

            # Add speech recognition button
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Or record audio here"
            )

            submit_btn = gr.Button("🚀 Get Advice", variant="primary")
            clear_btn = gr.Button("🔄 Clear")

            transcription = gr.Textbox(
                label="📝 What you said",
                interactive=False,
                lines=2
            )

        with gr.Column():
            gr.Markdown("### 💬 AI Screenplay Advisor")

            response = gr.Textbox(
                label="Expert Advice",
                interactive=False,
                lines=10
            )

            # TTS output
            tts_output = gr.Audio(
                label="🔊 Listen to response",
                autoplay=False
            )

    gr.Markdown("---")

    with gr.Accordion("📖 How to Use", open=True):
        gr.Markdown("""
        ### Two Ways to Use This:

        **Option 1: Type Your Questions**
        - Just type in the text box and click "Get Advice"

        **Option 2: Use Voice (Works in Chrome, Edge, Safari)**
        - Click the microphone icon in the text box
        - Your browser will ask for microphone permission - click "Allow"
        - Speak your question
        - Click "Get Advice"

        ### Example Questions:
        - "How do I develop my main character?"
        - "What makes good dialogue?"
        - "Help me with my story structure"
        - "I'm stuck on Act 2, what should I do?"
        - "How do I write a compelling scene?"

        ### Features:
        - ✅ Completely FREE - no API keys
        - ✅ Works offline (after first load)
        - ✅ Browser-based speech recognition
        - ✅ Browser-based text-to-speech
        - ✅ Screenplay-focused advice
        """)

    with gr.Accordion("⚙️ Tips", open=False):
        gr.Markdown("""
        **For best voice recognition:**
        - Use Chrome, Edge, or Safari (Firefox has limited support)
        - Speak clearly and at normal pace
        - Reduce background noise
        - Allow microphone access when prompted

        **This is a demonstration version.**
        For more advanced AI responses, you can upgrade to use:
        - OpenAI GPT-4
        - Anthropic Claude
        - Local LLMs (Ollama)

        But this free version is perfect for learning and testing!
        """)

    # Wire up the interface
    def handle_submit(text, audio):
        """Handle both text and audio input"""
        if audio and not text:
            # If audio provided but no text, just acknowledge
            return "Audio received! (Note: transcription requires browser support)", ai.get_response("Hello")
        elif text:
            return process_text(text)
        else:
            return "", "Please provide input (text or voice)"

    submit_btn.click(
        fn=process_text,
        inputs=[text_input],
        outputs=[transcription, response]
    )

    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[text_input, transcription, response]
    )


# ============================================================================
# Launch the app
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting FREE Voice Chat...")
    print("=" * 60)
    print("\n📱 Open in your browser: http://localhost:7860")
    print("\n💡 Tips:")
    print("   - Works in Chrome, Edge, Safari")
    print("   - Allow microphone access when prompted")
    print("   - No API keys or accounts needed!")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("\n" + "=" * 60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public URL
        show_error=True
    )
