# Getting Started - Simple Setup Guide

Don't worry! This guide will walk you through everything step-by-step. By the end, you'll have a working voice chat system.

## What You'll Need

1. **Python** (version 3.8 or higher)
   - Check if you have it: Open a terminal and type `python --version` or `python3 --version`
   - If you don't have it, download from: https://www.python.org/downloads/

2. **An OpenAI API Key** (easiest option to get started)
   - Go to: https://platform.openai.com/api-keys
   - Sign up or log in
   - Click "Create new secret key"
   - Copy the key somewhere safe (you'll need it in a minute)
   - Note: This will cost a small amount of money to use (usually a few cents per conversation)

3. **A microphone and speakers** (or headphones)

## Step-by-Step Setup

### Step 1: Get the Code

You already have this! You're in the right place.

### Step 2: Install Python Packages

Open a terminal in this folder and run:

```bash
pip install fastrtc gradio openai
```

Or if that doesn't work, try:

```bash
pip3 install fastrtc gradio openai
```

This will install everything you need. It might take a minute or two.

### Step 3: Set Your OpenAI API Key

You need to tell the program your API key. There are two ways:

**Option A: Environment Variable (Temporary - Easy)**

Every time before you run the program, run this command:

On Mac/Linux:
```bash
export OPENAI_API_KEY="your-key-here"
```

On Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=your-key-here
```

On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your-key-here"
```

Replace `your-key-here` with your actual API key.

**Option B: Create a .env File (Permanent - Better)**

1. Create a new file in this folder called `.env`
2. Add this line to it:
   ```
   OPENAI_API_KEY=your-key-here
   ```
3. Replace `your-key-here` with your actual API key
4. Save the file

Then install one more package:
```bash
pip install python-dotenv
```

### Step 4: Run the Simple Example

I've created a simple working version for you. Just run:

```bash
python voice_chat_simple.py
```

This will start a web interface at http://localhost:7860

Open that URL in your web browser!

### Step 5: Use It!

1. In your browser, you'll see a button that says "Start Voice Chat"
2. Click it
3. **Allow microphone access** when your browser asks
4. Start talking! Say something like "Hello, can you help me with my screenplay?"
5. When you stop talking, the system will:
   - Transcribe what you said
   - Send it to the AI
   - Speak the response back to you

## Troubleshooting

### "ModuleNotFoundError: No module named 'fastrtc'"

You need to install the packages. Run:
```bash
pip install fastrtc gradio openai
```

### "AuthenticationError: No API key provided"

You forgot to set your OpenAI API key. Go back to Step 3.

### "I can't hear any audio"

Check:
- Your speaker/headphone volume
- Your browser's site permissions (make sure audio is allowed)
- Try refreshing the page

### "The microphone isn't working"

Check:
- Your browser permissions (it should ask to use your microphone)
- Make sure you clicked "Start Voice Chat"
- Try refreshing the page and clicking "Allow" when asked

### "It's too slow"

The first response is always slower because it has to load. After that, it should be faster. If it's still slow:
- Make sure you have a good internet connection
- The AI and voice processing happen in the cloud, so some delay is normal

## What's Next?

Once you have this working, you can:

1. **Customize the AI's personality**: Edit the `screenplay_context` in the code to change how the AI responds
2. **Add screenplay-specific features**: Make it help with character development, plot structure, etc.
3. **Try local models**: Run everything on your computer (free, but more complex setup)

## Need More Help?

If you get stuck:
1. Read the error message carefully - it often tells you what's wrong
2. Make sure your API key is set correctly (Step 3)
3. Try running `pip install --upgrade fastrtc gradio openai` to update everything
4. Check that your Python version is 3.8 or higher: `python --version`

## Cost Information

Using OpenAI's APIs costs money, but it's very affordable for testing:

- **Whisper (Speech-to-Text)**: ~$0.006 per minute of audio
- **GPT-4 (AI responses)**: ~$0.01-0.03 per conversation turn
- **TTS (Text-to-Speech)**: ~$0.015 per 1000 characters

A typical conversation might cost 5-10 cents. You can set spending limits in your OpenAI account settings.

## Free Alternative (More Complex)

If you don't want to pay for APIs, you can run everything locally:
1. Install Whisper locally for speech-to-text
2. Use Ollama with a local AI model
3. Use Edge TTS (free) or pyttsx3 (offline) for voice

This is more complex to set up. If you want to try this, let me know and I can help!
