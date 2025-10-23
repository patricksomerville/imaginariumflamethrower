#!/bin/bash
# Installation script for Mac/Linux

echo "=========================================="
echo "Screenplay Voice Assistant - Installation"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.8 or higher from: https://www.python.org/downloads/"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed!"
    echo "Please install pip3"
    exit 1
fi

echo "✓ pip3 found"
echo ""

# Install required packages
echo "Installing required packages..."
echo "This may take a few minutes..."
echo ""

pip3 install --upgrade pip
pip3 install openai gradio python-dotenv

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    echo "✓ .env file found"
else
    echo "⚠️  No .env file found"
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "Please edit .env and add your OpenAI API key:"
    echo "  nano .env"
    echo ""
    echo "Or set it now:"
    read -p "Enter your OpenAI API key (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        echo "OPENAI_API_KEY=$api_key" > .env
        echo "✓ API key saved to .env"
    fi
fi

echo ""
echo "=========================================="
echo "Ready to run!"
echo "=========================================="
echo ""
echo "Start the voice assistant with:"
echo "  python3 voice_chat_simple.py"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:7860"
echo ""
