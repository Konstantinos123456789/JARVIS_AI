# Jarvis AI - Complete Setup Guide

This guide will walk you through setting up Jarvis AI from scratch.

## Step 1: Prerequisites

### Install Python
1. Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Verify installation:
   ```bash
   python --version
   ```

### Install Git (Optional, for cloning)
Download from [git-scm.com](https://git-scm.com/)

## Step 2: Download the Project

### Option A: Using Git
```bash
git clone https://github.com/yourusername/jarvis-ai.git
cd jarvis-ai
```

### Option B: Download ZIP
1. Download the ZIP file from GitHub
2. Extract to a folder
3. Open Command Prompt in that folder

## Step 3: Create Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

You should see `(venv)` in your terminal.

## Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This may take 10-15 minutes.

### Install PyAudio (Windows)

PyAudio can be tricky on Windows. If it fails:

1. Download the wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
   - Choose the file matching your Python version and system (cp38 = Python 3.8, cp39 = Python 3.9, etc.)
   - For 64-bit Python 3.9: `PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl`

2. Install it:
   ```bash
   pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
   ```

## Step 5: Download NLTK Data

Run Python and execute:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

## Step 6: Configure the Project

### 6.1 Create .env File
Copy the example file:
```bash
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

Edit `.env` with your information:
```env
NASA_API_KEY=your_actual_api_key_here
USERNAME=YourName
BOTNAME=Jarvis
```

### 6.2 Get NASA API Key
1. Go to [https://api.nasa.gov/](https://api.nasa.gov/)
2. Fill out the form
3. Check your email for the API key
4. Add it to your `.env` file

### 6.3 Edit config.py
Open `config.py` and verify the settings look correct.

### 6.4 Edit knowledge.py
Replace placeholders with your information:
- Line 10: Change `"User"` node attributes
- Lines 20-30: Add your personal information

Example:
```python
G.add_node("User", 
    type="Person", 
    name="John", 
    surname="Doe", 
    age='25', 
    birthday="1999-01-15"
)
```

### 6.5 Edit dates.py (Optional)
Add your special days in the `special_days` dictionary:
```python
special_days = {
    (12, 25): "Christmas",
    (1, 1): "New Year",
    (5, 15): "My Birthday",  # Add your days
    (6, 10): "Anniversary",
}
```

## Step 7: Test Microphone

Before running Jarvis, test your microphone:

```python
python -c "import speech_recognition as sr; r = sr.Recognizer(); print('Say something!'); \
with sr.Microphone() as source: audio = r.listen(source); \
print('You said:', r.recognize_google(audio))"
```

If this works, your microphone is ready!

## Step 8: Run Jarvis

```bash
python jarvis.py
```

### First Run Checklist
- ✅ Microphone is connected and working
- ✅ Speakers/headphones are connected
- ✅ Camera is connected (optional)
- ✅ Internet connection is active
- ✅ Virtual environment is activated

## Step 9: Test Commands

Try these commands to test:
1. "Hello" - Basic greeting
2. "What's the date" - Date check
3. "What's the time" - Time check
4. "Open calculator" - System command
5. "Tell me a joke" - Entertainment
6. "Exit" - Close Jarvis

## Common Issues & Solutions

### Issue: "ModuleNotFoundError"
**Solution:** Make sure virtual environment is activated and run:
```bash
pip install -r requirements.txt
```

### Issue: "Microphone not working"
**Solutions:**
1. Check Windows Privacy Settings → Microphone → Allow apps to access
2. Select correct microphone in Windows Sound Settings
3. Try running as Administrator

### Issue: "Could not understand audio"
**Solutions:**
1. Reduce background noise
2. Speak clearly and at normal pace
3. Adjust microphone volume in Windows settings
4. Move closer to microphone

### Issue: "NASA API not working"
**Solutions:**
1. Verify API key in `.env` file
2. Check internet connection
3. Try the DEMO_KEY first to test
4. Wait a few hours after registering for API key activation

### Issue: "Camera not detected"
**Solution:** The camera is optional. Jarvis will skip face detection if camera fails.

### Issue: "Speech is too fast/slow"
**Solution:** Edit `help.py` line 10:
```python
engine.setProperty('rate', 150)  # Lower = slower, Higher = faster
```

### Issue: "Wrong voice gender"
**Solution:** Edit `help.py` line 12:
```python
engine.setProperty('voice', voices[0].id)  # Try 0 or 1
```

## Optional Features

### Enable Ollama Integration (Advanced)
For the conversation feature, install Ollama:
1. Download from [ollama.ai](https://ollama.ai)
2. Install and run: `ollama run llama2`
3. Jarvis will now be able to have conversations

### Add More Commands
Edit `jarvis.py` and add your custom commands in the `main()` function.

## Updating the Project

To update to the latest version:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstalling

1. Deactivate virtual environment: `deactivate`
2. Delete the project folder
3. Done!

## Need Help?

- Check the main [README.md](README.md)
- Open an issue on GitHub
- Review error messages carefully

## Next Steps

Once everything works:
1. Customize your knowledge graph in `knowledge.py`
2. Add your favorite stocks in `config.py`
3. Personalize responses in `help.py`
4. Add custom voice commands in `jarvis.py`
5. Share your improvements on GitHub!

---

**Congratulations! You now have Jarvis AI running on your computer! 🎉**
