import pyautogui
import pyttsx3
import psutil
import os
import platform
import subprocess
import speech_recognition as sr
import pywhatkit
import wikipedia
import requests

# ── Platform detection ────────────────────────────────────────────────────────
OS = platform.system()  # 'Windows', 'Linux', 'Darwin'

# ── TTS Engine ────────────────────────────────────────────────────────────────
def _init_tts_engine():
    """Initialize pyttsx3 with the correct backend for each platform"""
    try:
        if OS == 'Windows':
            engine = pyttsx3.init('sapi5')
        elif OS == 'Darwin':
            engine = pyttsx3.init('nsss')   # macOS native
        else:
            engine = pyttsx3.init('espeak') # Linux
        engine.setProperty('rate', 190)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
        return engine
    except Exception as e:
        print(f"TTS init error: {e}")
        return pyttsx3.init()  # fallback to default

engine = _init_tts_engine()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Takes Input from User
def take_user_input():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('Listening....')
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print('Recognizing...')
        query = r.recognize_google(audio, language='en-in')
        query = query.lower()
        print(f"User said: {query}")
        
        for phrase, response in SMALL_TALK_PHRASES.items():
            if phrase in query:
                speak(response)
                return None

        return query
    
    except Exception as e:
        speak('Sorry, I could not understand. Could you please say that again?')
        return 'None'

def speak(text):
    """Speaks the given text using text-to-speech"""
    try:
        engine.stop()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")
    
def cpu() -> None:
    """Reports CPU and battery usage"""
    try:
        usage = str(psutil.cpu_percent())
        speak(f"CPU is at {usage} percent")
        battery = psutil.sensors_battery()
        if battery:
            speak(f"Battery is at {battery.percent} percent")
        else:
            speak("Battery information not available")
    except Exception as e:
        print(f"Error getting CPU info: {e}")
        speak("Sorry, I couldn't get system information")
    
def screenshot() -> None:
    """Takes a screenshot and saves it"""
    try:
        img = pyautogui.screenshot()
        folder_path = os.path.join(BASE_DIR, "Database", "Screenshots")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        img.save(os.path.join(folder_path, filename))
        speak("Screenshot saved successfully")
    except Exception as e:
        print(f"Screenshot error: {e}")
        speak("Sorry, I couldn't take a screenshot")


def open_calculator():
    """Opens the calculator application (cross-platform)"""
    try:
        if OS == 'Windows':
            os.system('calc')
        elif OS == 'Darwin':
            subprocess.Popen(['open', '-a', 'Calculator'])
        else:
            subprocess.Popen(['gnome-calculator'])
        speak("Opening calculator")
    except Exception as e:
        speak("Sorry, I couldn't open the calculator")
        print(f"Error: {e}")

def open_camera():
    """Opens the default camera application (cross-platform)"""
    try:
        if OS == 'Windows':
            os.system('start microsoft.windows.camera:')
        elif OS == 'Darwin':
            subprocess.Popen(['open', '-a', 'FaceTime'])
        else:
            subprocess.Popen(['cheese'])
        speak("Opening camera")
    except Exception as e:
        speak("Sorry, I couldn't open the camera")
        print(f"Error: {e}")

def open_cmd():
    """Opens a terminal (cross-platform)"""
    try:
        if OS == 'Windows':
            os.system('start cmd')
        elif OS == 'Darwin':
            subprocess.Popen(['open', '-a', 'Terminal'])
        else:
            subprocess.Popen(['gnome-terminal'])
        speak("Opening terminal")
    except Exception as e:
        speak("Sorry, I couldn't open the terminal")
        print(f"Error: {e}")

def open_notepad():
    """Opens a text editor (cross-platform)"""
    try:
        if OS == 'Windows':
            os.system('notepad')
        elif OS == 'Darwin':
            subprocess.Popen(['open', '-a', 'TextEdit'])
        else:
            subprocess.Popen(['gedit'])
        speak("Opening text editor")
    except Exception as e:
        speak("Sorry, I couldn't open the text editor")
        print(f"Error: {e}")

def play_on_youtube(video):
    """Plays a video on YouTube
    
    Args:
        video (str): The video name to search for
    """
    try:
        speak(f"Playing {video} on YouTube")
        pywhatkit.playonyt(video)
    except Exception as e:
        speak("Sorry, I couldn't play that video")
        print(f"Error: {e}")

def search_on_google(query):
    """Searches Google for the given query
    
    Args:
        query (str): The search query
    """
    try:
        speak(f"Searching Google for {query}")
        pywhatkit.search(query)
    except Exception as e:
        speak("Sorry, I couldn't perform the search")
        print(f"Error: {e}")

def search_on_wikipedia(query):
    """Searches Wikipedia and returns summary
    
    Args:
        query (str): The search query
        
    Returns:
        str: Wikipedia summary
    """
    try:
        results = wikipedia.summary(query, sentences=2)
        return results
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found. Please be more specific. Options include: {', '.join(e.options[:3])}"
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find that on Wikipedia"
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, there was an error searching Wikipedia"

def find_my_ip():
    """Finds and speaks the user's public IP address"""
    try:
        ip_address = requests.get('https://api.ipify.org').text
        speak(f"Your IP address is {ip_address}")
        print(f"IP Address: {ip_address}")
        return ip_address
    except Exception as e:
        speak("Sorry, I couldn't find your IP address")
        print(f"Error: {e}")
        return None


SMALL_TALK_PHRASES = {
    "how are you": "I'm just a computer program, but I'm functioning as intended. How can I help you?",
    "what's the weather like today": "I'm sorry, I don't have the ability to check the weather. Can I help you with something else?",
    "hi": "Hello! How can I assist you today?",
    "hello": "Hello! How can I help you?",
    "how is it going": "I'm just a computer program, but I'm ready to assist you. What do you need?",
    "goodbye": "Goodbye! Have a great day!",
    "bye": "Goodbye! Have a great day!",
    "what are you doing": "I am ready to help you. What would you like me to do?",
    "i'm bored": "I'm here to keep you entertained! What would you like to do?",
    "what can you do": "I can perform a variety of tasks, such as answering questions, playing music, opening applications, and more. Just let me know what you need.",
    "take a break": "Sure, I'll be here when you're ready to resume.",
    "i'm happy": "That's great to hear! What's making you feel that way?",
    "lets get to work": "Of course! Let's start working. Do you want to work on something specific?",
    "thanks jarvis": "You're welcome! I was made to be your helping assistant.",
    "thank you": "You're welcome! Happy to help.",
    "whats your name": "My name is Jarvis",
    "who made you": "I was created by my developer",
    "are you there": "Yes sir, I am here. How can I help you?",
}

def startup():
    """Startup sequence for Jarvis"""
    messages = [
        "Initializing Jarvis",
        "Starting all systems applications",
        "Installing and checking all drivers",
        "Calibrating and examining all the core processors",
        "Checking the internet connection",
        "Wait a moment sir",
        "All drivers are up and running",
        "All systems have been activated",
        "Now I am online"
    ]
    
    for msg in messages:
        speak(msg)
        print(msg)
    
    
def create_file():
    """Creates a new text file with user input"""
    try:
        speak("What should be the name of the file?")
        filename = take_user_input().lower()
        if filename == 'None':
            speak("I didn't catch that.")
            return
            
        folder_path = os.path.join(BASE_DIR, "Database", "NASA", "Images")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, f"{filename}.txt")
        
        speak("What content do you want to write in the file?")
        content = take_user_input().lower()
        if content == 'None':
            speak("I didn't catch that.")
            return
            
        with open(file_path, "w") as file:
            file.write(content)
        speak(f"File {filename}.txt has been created.")
    except Exception as e:
        print(f"File creation error: {e}")
        speak("Sorry, I couldn't create the file")