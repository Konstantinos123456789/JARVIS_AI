"""
Configuration file for Jarvis AI
IMPORTANT: Edit this file with your personal information and API keys
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Personal Information
USERNAME = os.getenv("USERNAME", "YOUR_NAME")
BOTNAME = os.getenv("BOTNAME", "Jarvis")

# API Keys - CHANGE THESE!
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")  # Get your key from https://api.nasa.gov/

# Display Settings
SET_WIDTH = 800
SET_HEIGHT = 700

# Voice Settings
VOICE_RATE = 190  # Speech rate (words per minute)
VOICE_VOLUME = 1.0  # Volume (0.0 to 1.0)
VOICE_INDEX = 1  # Voice selection (0 for male, 1 for female - varies by system)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "Database")
SCREENSHOTS_PATH = os.path.join(DATABASE_PATH, "Screenshots")
NOTEPAD_PATH = os.path.join(DATABASE_PATH, "Notepad")
NASA_IMAGES_PATH = os.path.join(DATABASE_PATH, "NASA", "Images")

# Create directories if they don't exist
for path in [DATABASE_PATH, SCREENSHOTS_PATH, NOTEPAD_PATH, NASA_IMAGES_PATH]:
    os.makedirs(path, exist_ok=True)

# Stock Symbols
STOCK_SYMBOLS = [
    'AAPL', 'GOOG', 'MSFT', 'AMZN', 
    'NVDA', 'TSLA', 'INTC', 'CSCO',
    'JPM', 'BAC', 'AXP', 'KO', 'NKE'
]

# Personal Information for Knowledge Graph
# Edit these in knowledge.py directly or here
PERSONAL_INFO = {
    "name": "YOUR_NAME",
    "surname": "YOUR_SURNAME",
    "age": "YOUR_AGE",
    "birthday": "YOUR_BIRTHDAY",  # Format: "YYYY-MM-DD"
    "location": "YOUR_CITY",
    "country": "YOUR_COUNTRY"
}
