#!/bin/bash
# Simple one-command setup and run script

echo "=================================="
echo "Voice Chat Quick Start"
echo "=================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python first."
    exit 1
fi

echo "✓ Python found"

# Install packages
echo "Installing packages..."
pip3 install --quiet openai gradio python-dotenv 2>/dev/null || {
    echo "⚠️  Using pip instead of pip3..."
    pip install --quiet openai gradio python-dotenv
}

echo "✓ Packages installed"
echo ""

# Check for API key
if [ ! -f ".env" ]; then
    echo "❌ No .env file found!"
    echo ""
    echo "Please create a .env file with your OpenAI API key:"
    echo "  echo 'OPENAI_API_KEY=your-key-here' > .env"
    echo ""
    echo "Get your API key from: https://platform.openai.com/api-keys"
    exit 1
fi

echo "✓ .env file found"
echo ""

# Run the app
echo "=================================="
echo "Starting Voice Chat App..."
echo "=================================="
echo ""
echo "The app will open at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 voice_chat_simple.py
