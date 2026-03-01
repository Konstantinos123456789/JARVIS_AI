"""
Configuration file for Jarvis AI
Copy .env.example to .env and fill in your values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Personal 
USERNAME = os.getenv("JARVIS_USERNAME", "User")
BOTNAME  = os.getenv("BOTNAME", "Jarvis")

# API Keys 
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

# Display 
SET_WIDTH  = 800
SET_HEIGHT = 700

# Voice 
VOICE_RATE   = 190
VOICE_VOLUME = 1.0
VOICE_INDEX  = 1

# Paths 
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH    = os.path.join(BASE_DIR, "Database")
SCREENSHOTS_PATH = os.path.join(DATABASE_PATH, "Screenshots")
NOTEPAD_PATH     = os.path.join(DATABASE_PATH, "Notepad")
NASA_IMAGES_PATH = os.path.join(DATABASE_PATH, "NASA", "Images")
STOCK_DATA_PATH  = os.path.join(DATABASE_PATH, "StockData")
MOVIELENS_PATH   = os.path.join(DATABASE_PATH, "MovieLens", "ml-latest-small")

for path in [DATABASE_PATH, SCREENSHOTS_PATH, NOTEPAD_PATH, NASA_IMAGES_PATH, STOCK_DATA_PATH]:
    os.makedirs(path, exist_ok=True)

# Stock Symbols 
STOCK_SYMBOLS = [
    'AAPL', 'GOOG', 'MSFT', 'AMZN',
    'NVDA', 'TSLA', 'INTC', 'CSCO',
    'JPM',  'BAC',  'AXP',  'KO', 'NKE'
]