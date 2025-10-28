#!/usr/bin/env python3
"""
Super simple test - just verifies OpenAI API works
Run this first to make sure everything is set up correctly
"""

import os
import sys

print("=" * 60)
print("Testing Your Setup")
print("=" * 60)
print()

# Step 1: Check Python version
print("1. Checking Python version...")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 8:
    print(f"   ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"   ✗ Python {python_version.major}.{python_version.minor} is too old")
    print("   Please install Python 3.8 or higher")
    sys.exit(1)

print()

# Step 2: Check for packages
print("2. Checking for required packages...")

try:
    import openai
    print("   ✓ openai installed")
except ImportError:
    print("   ✗ openai not installed")
    print("   Run: pip install openai")
    sys.exit(1)

try:
    import gradio
    print("   ✓ gradio installed")
except ImportError:
    print("   ✗ gradio not installed")
    print("   Run: pip install gradio")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("   ✓ python-dotenv installed")
except ImportError:
    print("   ⚠  python-dotenv not installed (optional)")
    print("   Run: pip install python-dotenv")

print()

# Step 3: Check for API key
print("3. Checking for OpenAI API key...")

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("   ✗ No API key found!")
    print()
    print("   Please set your API key:")
    print("   1. Create a .env file with: OPENAI_API_KEY=your-key")
    print("   2. Or run: export OPENAI_API_KEY='your-key'")
    print()
    print("   Get your key from: https://platform.openai.com/api-keys")
    sys.exit(1)

if api_key.startswith("sk-"):
    print(f"   ✓ API key found (starts with: {api_key[:8]}...)")
else:
    print("   ⚠  API key found but doesn't look right")
    print(f"   It should start with 'sk-' but starts with: {api_key[:8]}")

print()

# Step 4: Test API connection
print("4. Testing OpenAI API connection...")
print("   (This may take a few seconds...)")

try:
    client = openai.OpenAI()

    # Try a simple API call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'Hello!'"}],
        max_tokens=10
    )

    result = response.choices[0].message.content
    print(f"   ✓ API working! Response: {result}")

except Exception as e:
    print(f"   ✗ API test failed: {e}")
    print()
    print("   Common issues:")
    print("   - Invalid API key")
    print("   - No internet connection")
    print("   - OpenAI API is down")
    print("   - No credits in your OpenAI account")
    sys.exit(1)

print()
print("=" * 60)
print("SUCCESS! Everything is set up correctly! ✓")
print("=" * 60)
print()
print("You can now run the voice chat app:")
print("  python3 voice_chat_simple.py")
print()
print("Or use the quick start script:")
print("  ./quick-start.sh")
print()
