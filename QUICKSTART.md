# Quick Start - Get Running in 2 Minutes! 🚀

**You already have everything set up!** Just follow these 3 steps:

## Step 1: Install Packages (1 minute)

Open a terminal in this folder and run:

### On Mac/Linux:
```bash
chmod +x install.sh
./install.sh
```

### On Windows:
```cmd
install.bat
```

### Or manually:
```bash
pip install openai gradio python-dotenv
```

## Step 2: Run the App (10 seconds)

```bash
python voice_chat_simple.py
```

Or on some systems:
```bash
python3 voice_chat_simple.py
```

## Step 3: Open in Browser

Go to: **http://localhost:7860**

## That's It!

You should see a web interface. Click the microphone button, allow browser access, and start talking!

## First Time Using?

Try saying:
- "Hello, can you help me with my screenplay?"
- "How do I structure a good plot?"
- "Help me develop my main character"

The AI will respond with text AND voice!

---

## Troubleshooting

### "No module named 'openai'"
Run: `pip install openai gradio python-dotenv`

### "No API key provided"
Your API key should already be in the `.env` file. If not:
1. Open `.env` file
2. Make sure it has: `OPENAI_API_KEY=your-key-here`
3. Save and try again

### Can't hear audio
- Check your volume
- Make sure browser allows audio
- Try clicking the play button on the audio player

---

**Need more help?** See `GETTING_STARTED.md` for detailed instructions.
