# Jarvis AI - Personal Voice Assistant

A Python-based AI voice assistant inspired by Iron Man's JARVIS, featuring voice recognition, natural language processing, computer vision, and various automation capabilities.

## ⚠️ Important Notice

**This project is currently Windows-only** due to dependencies on Windows-specific features (taskkill, os.startfile, etc.). Cross-platform support is planned for future versions.

## Features

- 🎤 **Voice Recognition** - Hands-free control using speech
- 🔊 **Text-to-Speech** - Natural voice responses
- 👁️ **Face Detection** - Recognizes when you're present
- 🧠 **Emotion Detection** - Analyzes sentiment from your speech
- 📊 **Stock Recommendations** - AI-powered financial advice
- 🌌 **NASA Integration** - Fetch space news and asteroid data
- 🌐 **Web Automation** - Google, YouTube, Wikipedia searches
- 📝 **Note Taking** - Voice-controlled note creation
- 🗓️ **Special Days Tracking** - Remember important dates
- 🎮 **Browser Automation** - Chrome tab control
- 🧮 **Calculator** - Voice-activated calculations
- 💾 **Personal Knowledge Graph** - Stores and retrieves personal information

## Prerequisites

### System Requirements
- **OS:** Windows 10/11 (64-bit)
- **Python:** 3.8 or higher
- **RAM:** Minimum 8GB (16GB recommended for ML models)
- **Storage:** 5GB free space

### Required System Software
- **Microphone** - For voice input
- **Webcam** - For face detection (optional)
- **Internet Connection** - For web features and API calls

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/jarvis-ai.git
cd jarvis-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

### 5. Create Required Directories
```bash
mkdir Database
mkdir Database\Screenshots
mkdir Database\Notepad
mkdir Database\NASA
mkdir Database\NASA\Images
```

### 6. Configure Environment Variables

Create a `.env` file in the project root:
```env
NASA_API_KEY=your_nasa_api_key_here
USERNAME=YourName
BOTNAME=Jarvis
```

Get a free NASA API key from: https://api.nasa.gov/

### 7. Update Personal Information

Edit `knowledge.py` and replace placeholders:
- `YOUR_NAME` - Your actual name
- `YOUR_SURNAME` - Your surname
- `YOUR_AGE` - Your age
- `YOUR_BIRTHDAY` - Your birthday
- `YOUR_LOCATION` - Your city
- `YOUR_COUNTRY` - Your country

## Usage

### Starting Jarvis
```bash
python jarvis.py
```

### Voice Commands Examples

**General:**
- "Hello" / "Hi"
- "How are you"
- "What's the date"
- "What's the time"
- "Exit" / "Sleep"

**Web & Search:**
- "Search Wikipedia for [topic]"
- "Play [video name] on YouTube"
- "Search Google for [query]"
- "Open GitHub"
- "Open Stack Overflow"

**System Control:**
- "Open calculator"
- "Open camera"
- "Open notepad"
- "Take a screenshot"
- "What's my CPU usage"

**Notes & Memory:**
- "Write a note"
- "Remember that [something]"
- "Do you remember anything"
- "Create file"

**Personal Information:**
- "How old am I"
- "What is my favorite movie"
- "Where do I live"
- "What is my favorite cuisine"
- "What is my favorite book"

**Space & Science:**
- "Space news"
- "Summary about [astronomical term]"
- "Asteroid information"

**Finance:**
- "Give me stock recommendations" (then specify your investment goals)

**Entertainment:**
- "Tell me a joke"

**Browser Automation:**
- "Google automation new tab"
- "Google automation close tab"
- "Google automation switch tab 2"

**Special Days:**
- "Add special day"

## Configuration

### Adjusting Voice Settings
Edit `help.py` to change voice properties:
```python
engine.setProperty('rate', 190)  # Speech rate
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
engine.setProperty('voice', voices[1].id)  # Voice selection (0 or 1)
```

### Adding More Knowledge
Edit `knowledge.py` to add nodes and edges to the knowledge graph:
```python
G.add_node("NodeName", type="Type", attribute="value")
G.add_edge("Node1", "Node2", type="relationship")
```

## Project Structure

```
jarvis-ai/
│
├── jarvis.py              # Main entry point
├── help.py                # Helper functions (speak, take_user_input, etc.)
├── knowledge.py           # Knowledge graph for personal data
├── data.py                # Text classification (needs data)
├── dates.py               # Special days tracking
├── stock.py               # Stock recommendation system
├── gesture_recognition.py # Face detection and gestures
├── utils.py               # Utility constants
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (create this)
└── Database/              # Data storage (auto-created)
    ├── Screenshots/
    ├── Notepad/
    └── NASA/
        └── Images/
```

## Known Issues

1. **Missing Functions:** Some imported functions in `jarvis.py` are not yet implemented in `help.py`:
   - `open_calculator()`, `open_camera()`, `open_cmd()`, `open_notepad()`
   - `play_on_youtube()`, `search_on_google()`, `search_on_wikipedia()`, `find_my_ip()`

2. **Ollama Integration:** The `talk_to_user()` function requires Ollama to be running locally (currently commented out)

3. **Windows-Only:** Uses Windows-specific commands that won't work on Linux/Mac

4. **Data Classification:** `data.py` needs actual training data

## Troubleshooting

### "Could not understand" Error
- Check microphone permissions
- Ensure microphone is working and not muted
- Reduce background noise
- Speak clearly and at normal pace

### Import Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+)

### NASA API Errors
- Verify API key in `.env` file
- Check internet connection
- Ensure you haven't exceeded API rate limits

### Face Detection Not Working
- Check webcam permissions
- Ensure webcam is connected and working
- Try running as administrator

## Future Enhancements

- [ ] Cross-platform compatibility (Linux, macOS)
- [ ] Implement missing helper functions
- [ ] Add GUI interface
- [ ] Cloud sync for knowledge graph
- [ ] Multi-language support
- [ ] Integration with smart home devices
- [ ] Mobile app companion
- [ ] Voice customization
- [ ] Advanced gesture controls

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Marvel's JARVIS from Iron Man
- Built with Python and various open-source libraries
- NASA API for space data
- Google Speech Recognition for voice input

## Disclaimer

This is a personal project created for educational purposes. Use responsibly and at your own risk. The stock recommendation feature is for informational purposes only and should not be considered financial advice.


Project Link: [https://github.com/Konstantinos123456789/JARVIS_AI](https://github.com/Konstantinos123456789/JARVIS_AI)
