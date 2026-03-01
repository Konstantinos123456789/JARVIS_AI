"""
Jarvis AI — single entry point
  python jarvis.py          → GUI mode  (default)
  python jarvis.py --cli    → CLI mode  (terminal only)
"""

import os
import platform
import random
import sys
import threading
import webbrowser
from datetime import datetime, timedelta

import cv2
import numpy as np
import pyjokes
import psutil
import pyttsx3
import requests
import torch
from PIL import Image
from keyboard import press_and_release
from transformers import BertModel, BertTokenizer

import knowledge
from config import BASE_DIR, BOTNAME, NASA_API_KEY, SET_WIDTH, SET_HEIGHT, USERNAME
from dates import add_special_day, check_special_days
from help import (SMALL_TALK_PHRASES, cpu, create_file, find_my_ip,
                  open_calculator, open_camera, open_cmd, open_notepad,
                  play_on_youtube, screenshot, search_on_google,
                  search_on_wikipedia, speak, startup, take_user_input)
from stock import generate_recommendations

# ── Platform & OpenCV ──────────────────────────────────────────────────────────
OS = platform.system()
cv2.setLogLevel(3)  # suppress obsensor camera probe errors

# ── BERT ───────────────────────────────────────────────────────────────────────
tokenizer      = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model     = BertModel.from_pretrained('bert-base-uncased')
EMOTION_LABELS = ['happiness', 'sadness', 'neutral']


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CORE LOGIC  (shared by GUI and CLI)
# ═══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


class GestureRecognizer:
    def __init__(self):
        self.gesture_model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.gesture_model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def recognize_gesture(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area         = cv2.contourArea(contour)
            x, y, w, h   = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            features     = np.array([[area, aspect_ratio]], dtype=np.float32)
            _, results, _, _ = self.gesture_model.findNearest(features, k=5)
            gesture_map  = {1: "Thumbs up!", 2: "Waving!", 3: "Pointing!"}
            if results[0] in gesture_map:
                return gesture_map[results[0]]
        return "Unknown gesture"


def greet_user(say=speak) -> None:
    hour     = datetime.now().hour
    greeting = ("Good Morning"   if 6  <= hour < 12 else
                "Good Afternoon" if 12 <= hour < 16 else
                "Good Evening")
    msg = f"{greeting} {USERNAME}. I am {BOTNAME}. How may I assist you?"
    say(msg)
    print(msg)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                faces = FaceDetector().detect_faces(frame)
                say("I am able to see you!" if len(faces) > 0 else "I don't see anyone")
            cap.release()
        else:
            print("No camera available, skipping face detection.")
    except Exception as e:
        print(f"Camera error: {e}")


def detect_emotions(say=speak) -> str:
    try:
        user_input = take_user_input()
        if user_input == 'None':
            return "I couldn't understand what you said."
        inputs  = tokenizer.encode_plus(
            user_input, add_special_tokens=True, max_length=512,
            return_attention_mask=True, return_tensors='pt', truncation=True
        )
        outputs       = bert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        scores        = outputs.last_hidden_state[:, 0, :]
        idx           = torch.argmax(scores).item()
        emotion_index = idx if 0 <= idx < len(EMOTION_LABELS) else 2
        emotion       = EMOTION_LABELS[emotion_index]
        score         = int(scores[0][emotion_index].item())
        return f"I sense that you're feeling {emotion} with a score of {score}."
    except Exception as e:
        print(f"Error detecting emotions: {e}")
        return "I'm having trouble analyzing your emotions right now."


def calculate(say=speak) -> None:
    try:
        say("What operation do you want to make?")
        operation = take_user_input()
        if not operation or operation == 'None':
            say("I didn't catch that.")
            return
        say("Enter the first number")
        num1_str = take_user_input()
        say("Enter the second number")
        num2_str = take_user_input()
        try:
            num1, num2 = float(num1_str), float(num2_str)
        except (ValueError, TypeError):
            say("Sorry, I need valid numbers.")
            return
        operations = {
            "addition":       num1 + num2,
            "add":            num1 + num2,
            "subtraction":    num1 - num2,
            "subtract":       num1 - num2,
            "multiplication": num1 * num2,
            "multiply":       num1 * num2,
            "division":       num1 / num2 if num2 != 0 else 'undefined',
        }
        result = operations.get(operation, "Operation not supported")
        say(f"The result is {result}")
    except Exception as e:
        print(f"Calculation error: {e}")
        say("Sorry, there was an error with the calculation.")


def notepad_write(say=speak) -> str | None:
    say("Tell me the query. I am ready to write.")
    text = take_user_input()
    if not text or text == 'None':
        say("I didn't catch that, please try again.")
        return None
    folder   = os.path.join(BASE_DIR, "Database", "Notepad")
    os.makedirs(folder, exist_ok=True)
    filename = f"{datetime.now().strftime('%H-%M')}-note.txt"
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        f.write(text)
    say(f"Note saved as {filename}")
    return filename


def close_notepad(say=speak) -> None:
    say("Closing notepad.")
    if OS == 'Windows':
        os.system('taskkill /f /im notepad.exe')
    elif OS == 'Darwin':
        os.system('pkill -x TextEdit')
    else:
        os.system('pkill gedit')


def chrome_automation(command: str) -> None:
    key_mappings = {
        'new tab':    'ctrl+t',
        'close tab':  'ctrl+w',
        'new window': 'ctrl+n',
        'history':    'ctrl+h',
        'download':   'ctrl+j',
    }
    for action, keys in key_mappings.items():
        if action in command:
            press_and_release(keys)
            return
    if 'switch tab' in command:
        tab_num = command.replace("switch tab", "").replace("to", "").strip()
        press_and_release(f'ctrl+{tab_num}')


def _talk_to_ollama(message: str) -> dict | None:
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama2", "messages": [{"role": "user", "content": message}], "stream": False}
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ollama error: {e}")
        return None


def talk_to_user(say=speak) -> None:
    say("What do you want to talk about?")
    topic = take_user_input()
    if topic and topic != 'None':
        responses = []
        t = threading.Thread(target=lambda: responses.append(_talk_to_ollama(topic)))
        t.start()
        t.join()
        if responses and responses[0]:
            try:
                say(responses[0]['message']['content'])
            except (KeyError, TypeError):
                say("I'm having trouble connecting to the conversation system.")
        else:
            say("Make sure Ollama is running.")


def nasa_news(date_param, say=speak) -> str:
    try:
        data      = requests.get(
            f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}&date={date_param}"
        ).json()
        title     = data.get('title', 'Astronomy Picture of the Day')
        info      = data.get('explanation', 'No explanation available.')
        image_url = data.get('url', '')
        if not image_url:
            say("No image available today.")
            return "No image available."
        folder    = os.path.join(BASE_DIR, "Database", "NASA", "Images")
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"{date_param}.jpg")
        with open(file_path, 'wb') as f:  # lgtm[py/clear-text-storage-sensitive-data]
            f.write(requests.get(image_url).content)
        Image.open(file_path).show()
        say(f"Title: {title}")
        say(f"According to NASA: {info}")
        return f"{title} — {info[:120]}..."
    except Exception as e:
        print(f"NASA error: {e}")
        say("Sorry, I couldn't fetch the news from NASA.")
        return "Error fetching NASA news."


def summary(body: str, say=speak) -> str:
    try:
        folder = os.path.join(BASE_DIR, "Database", "NASA", "Images")
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            if images:
                Image.open(os.path.join(folder, random.choice(images))).show()
        data = requests.get(f"https://hubblesite.org/api/v3/glossary/{body}").json()
        text = data.get('definition', 'No definition found.') if data else 'No data available.'
        say(f"According to NASA: {text}")
        return text
    except Exception as e:
        say("Sorry, I couldn't fetch the summary.")
        print(f"Summary error: {e}")
        return "Error fetching summary."


def astro(start_date, end_date, say=speak) -> str:
    try:
        data  = requests.get(
            f"https://api.nasa.gov/neo/rest/v1/feed"
            f"?start_date={start_date}&end_date={end_date}&api_key={NASA_API_KEY}"
        ).json()
        total = data['element_count']
        say(f"Total asteroids between {start_date} and {end_date} is: {total}")
        say("Extracting data for these asteroids.")
        for body in data['near_earth_objects'].get(str(start_date), []):
            say(f"Asteroid {body['name']} with id {body['id']} has magnitude {body['absolute_magnitude_h']}.")
        return f"{total} asteroids found."
    except Exception as e:
        print(f"Asteroid error: {e}")
        say("Sorry, I couldn't fetch the asteroid data.")
        return "Error fetching asteroid data."


def on_closing(say=speak) -> None:
    hour = datetime.now().hour
    say("Good night sir, take care!" if hour >= 21 or hour < 6 else "Have a good day sir!")


# ── Sentinel strings returned by process_query for multi-turn follow-ups ───────
_AWAIT_MEMORY = "__AWAIT_MEMORY__"
_AWAIT_STOCK  = "__AWAIT_STOCK__"
_EXIT         = "__EXIT__"


def process_query(query: str, say=speak, joke_state: dict = None) -> str | None:
    """
    Single command dispatcher shared by GUI and CLI.

    Returns:
      str   — response text to display / speak
      None  — action was fire-and-forget (already handled internally)
      _AWAIT_MEMORY / _AWAIT_STOCK — caller must collect a follow-up input
      _EXIT — caller should shut down
    """
    if joke_state is None:
        joke_state = {'told': False}

    q = query.lower()

    # ── Apps ───────────────────────────────────────────────────────────────────
    if 'open notepad' in q:
        open_notepad();  return "Opening text editor."
    if 'open command prompt' in q or 'open cmd' in q:
        open_cmd();      return "Opening terminal."
    if 'open camera' in q:
        open_camera();   return "Opening camera."
    if 'open calculator' in q:
        open_calculator(); return "Opening calculator."

    # ── Web search ─────────────────────────────────────────────────────────────
    if 'ip address' in q:
        ip = find_my_ip()
        return f"Your IP address is {ip}"

    if 'wikipedia' in q:
        topic = q.replace("wikipedia", "").replace("search", "").strip()
        if topic:
            result = search_on_wikipedia(topic)
            say(f"According to Wikipedia, {result}")
            return result
        say("What do you want to search on Wikipedia?")
        return None

    if 'youtube' in q:
        video = q.replace("youtube", "").replace("play", "").strip()
        if video:
            play_on_youtube(video)
            return f"Playing {video} on YouTube."
        say("What do you want to play on YouTube?")
        return None

    if 'search on google' in q:
        sq = q.replace("search on google", "").strip()
        search_on_google(sq)
        return f"Searching Google for '{sq}'"

    # ── Date / time ────────────────────────────────────────────────────────────
    if 'date' in q:
        result = f"The current date is {datetime.now().strftime('%d/%m/%Y')}"
        say(result);  return result

    if 'time' in q:
        result = f"Sir, the time is {datetime.now().strftime('%H:%M:%S')}"
        say(result);  return result

    # ── System ─────────────────────────────────────────────────────────────────
    if 'cpu' in q:
        usage   = psutil.cpu_percent()
        battery = psutil.sensors_battery()
        result  = f"CPU at {usage}%" + (f"  |  Battery at {battery.percent}%" if battery else "")
        say(result);  return result

    if 'screenshot' in q:
        screenshot()
        return "Screenshot saved."

    if 'speed of internet' in q or 'internet speed' in q:
        webbrowser.open("https://fast.com")
        return "Opening fast.com."

    if 'open github' in q:
        webbrowser.open("https://github.com")
        return "Opening GitHub."

    if 'open stackoverflow' in q or 'stack overflow' in q:
        webbrowser.open('https://stackoverflow.com')
        return "Opening Stack Overflow."

    if 'shutdown' in q:
        on_closing(say)
        if OS == 'Windows':
            os.system("shutdown /s /t 1")
        elif OS == 'Darwin':
            os.system("sudo shutdown -h now")
        else:
            os.system("shutdown -h now")
        return _EXIT

    # ── Fun ────────────────────────────────────────────────────────────────────
    if 'joke' in q or 'tell me a joke' in q:
        if not joke_state['told']:
            joke = pyjokes.get_joke()
            say(joke)
            joke_state['told'] = True
            return joke
        result = "You've already heard a joke. Would you like to hear another one?"
        say(result);  return result

    # ── Knowledge graph ────────────────────────────────────────────────────────
    if 'how old am i' in q or 'my age' in q:
        age    = knowledge.get_person_age(USERNAME)
        result = f"You are {age} years old." if age else "I don't have your age information."
        say(result);  return result

    if 'favorite movie' in q:
        movie  = knowledge.get_favorite_movie(USERNAME)
        result = f"Your favorite movie is {movie}." if movie else "I don't have your favorite movie information."
        say(result);  return result

    if 'where do i live' in q or 'my location' in q:
        city   = knowledge.get_person_location(USERNAME)
        result = f"You live in {city}." if city else "I don't have your location information."
        say(result);  return result

    if 'favorite cuisine' in q:
        cuisines = knowledge.get_favorite_cuisines(USERNAME)
        if cuisines and len(cuisines) >= 2:
            result = f"Your favorite cuisines are {cuisines[0]} and {cuisines[1]}."
        elif cuisines:
            result = f"Your favorite cuisine is {cuisines[0]}."
        else:
            result = "I don't have your favorite cuisine information."
        say(result);  return result

    if 'favorite book' in q:
        book   = knowledge.get_favorite_book(USERNAME)
        result = (f"Your favorite book is {book[0]} by {book[1]} in the {book[2]} genre."
                  if book else "I don't have your favorite book information.")
        say(result);  return result

    if 'favorite food' in q:
        foods  = knowledge.get_favorite_food(USERNAME)
        result = ("Your favorite foods are " + ", ".join(foods)
                  if foods else "I don't have your favorite food information.")
        say(result);  return result

    # ── AI / Emotions ──────────────────────────────────────────────────────────
    if 'how do i feel' in q or 'my emotion' in q:
        result = detect_emotions(say)
        say(result);  return result

    if 'talk' in q:
        talk_to_user(say);  return None

    # ── NASA ───────────────────────────────────────────────────────────────────
    if 'space news' in q or 'nasa news' in q:
        return nasa_news(datetime.now().date(), say)

    if 'summary' in q:
        body = q.replace("jarvis", "").replace("about", "").replace("summary", "").strip()
        if body:
            return summary(body, say)
        say("What would you like a summary about?")
        return None

    if 'asteroid' in q:
        return astro(datetime.now().date() - timedelta(days=30), datetime.now().date(), say)

    # ── Stocks (needs follow-up) ───────────────────────────────────────────────
    if 'stock recommendation' in q:
        say("What are your investing goals?")
        return _AWAIT_STOCK

    # ── Chrome ─────────────────────────────────────────────────────────────────
    if 'google automation' in q or 'chrome automation' in q:
        chrome_automation(q)
        return "Chrome automation executed."

    # ── Notes ──────────────────────────────────────────────────────────────────
    if 'write a note' in q:
        filename = notepad_write(say)
        return f"Note saved as {filename}." if filename else None

    if 'dismiss that note' in q or 'close notepad' in q:
        close_notepad(say);  return None

    # ── Memory (needs follow-up) ───────────────────────────────────────────────
    if 'remember that' in q:
        say("What should I remember sir?")
        return _AWAIT_MEMORY

    if 'do you remember' in q:
        try:
            with open(os.path.join(BASE_DIR, 'data.txt'), 'r') as f:
                content = f.read()
            result = f"You told me to remember that {content}"
            say(result);  return result
        except FileNotFoundError:
            result = "You haven't asked me to remember anything yet."
            say(result);  return result

    # ── Misc ───────────────────────────────────────────────────────────────────
    if 'calculation' in q or 'calculate' in q:
        calculate(say);  return None

    if 'create file' in q:
        create_file();  return None

    if 'add special day' in q:
        add_special_day();  return None

    if 'exit' in q or 'sleep' in q or 'goodbye' in q:
        on_closing(say);  return _EXIT

    result = "I'm not sure how to help with that. Could you try rephrasing?"
    say(result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLI MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_cli() -> None:
    startup()
    greet_user()
    check_special_days()
    joke_state = {'told': False}

    while True:
        query = take_user_input()
        if not query:
            continue

        result = process_query(query, say=speak, joke_state=joke_state)

        if result == _EXIT:
            break

        elif result == _AWAIT_MEMORY:
            msg = take_user_input()
            if msg and msg != 'None':
                with open(os.path.join(BASE_DIR, 'data.txt'), 'w') as f:
                    f.write(msg)
                speak(f"I'll remember that: {msg}")

        elif result == _AWAIT_STOCK:
            goals = take_user_input()
            if goals and goals != 'None':
                speak("Analyzing the stocks, please wait.")
                stocks = generate_recommendations(goals)
                speak(f'Recommended stocks: {", ".join(set(stocks))}'
                      if stocks else "No suitable recommendations found.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GUI MODE
# ═══════════════════════════════════════════════════════════════════════════════

import tkinter as tk
from tkinter import scrolledtext

# ── Theme ──────────────────────────────────────────────────────────────────────
BG        = "#020c14"
PANEL     = "#041525"
BORDER    = "#0d4f7c"
ACCENT    = "#00c8ff"
GREEN     = "#00ffaa"
RED       = "#ff3860"
TEXT      = "#c8e8f8"
TEXT_DIM  = "#3a6a8a"

FONT_TITLE = ("Courier", 11, "bold")
FONT_MONO  = ("Courier", 10)
FONT_SMALL = ("Courier", 8)
FONT_LOG   = ("Courier", 9)


def speak_async(text: str) -> None:
    """Thread-safe TTS — creates a fresh engine per call to avoid tkinter loop conflict."""
    def _speak():
        try:
            if OS == 'Windows':
                eng = pyttsx3.init('sapi5')
            elif OS == 'Darwin':
                eng = pyttsx3.init('nsss')
            else:
                eng = pyttsx3.init('espeak')
            eng.setProperty('rate', 190)
            eng.setProperty('volume', 1.0)
            voices = eng.getProperty('voices')
            eng.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
            eng.say(text)
            eng.runAndWait()
            eng.stop()
        except Exception as e:
            print(f"TTS error: {e}")
    threading.Thread(target=_speak, daemon=True).start()


class JarvisGUI:
    def __init__(self, root: tk.Tk):
        self.root            = root
        self.listening       = False
        self.joke_state      = {'told': False}
        self._pending_memory = False
        self._pending_stock  = False

        self.root.title(f"J.A.R.V.I.S  —  {BOTNAME}")
        self.root.geometry(f"{SET_WIDTH}x{SET_HEIGHT}")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._build_ui()
        self._start_clock()
        self._animate_status()
        self.root.after(200, self._startup_sequence)

    # ── Startup ────────────────────────────────────────────────────────────────

    def _startup_sequence(self):
        self.log_message("SYSTEM", "Initializing all systems...", color=TEXT_DIM)
        self.log_message("SYSTEM", f"Welcome, {USERNAME}. All systems online.", color=GREEN)
        self.log_message("SYSTEM", f"I am {BOTNAME}. How may I assist you?", color=ACCENT)
        threading.Thread(target=self._greet_thread, daemon=True).start()

    def _greet_thread(self):
        greet_user(say=speak_async)
        check_special_days()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_body()
        self._build_footer()

    def _build_header(self):
        header = tk.Frame(self.root, bg=PANEL, height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text="◈ J.A.R.V.I.S", bg=PANEL, fg=ACCENT,
                 font=("Courier", 14, "bold")).pack(side=tk.LEFT, padx=16, pady=12)
        tk.Label(header, text="JUST A RATHER VERY INTELLIGENT SYSTEM",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(side=tk.LEFT)
        self.clock_label = tk.Label(header, text="", bg=PANEL, fg=ACCENT, font=FONT_TITLE)
        self.clock_label.pack(side=tk.RIGHT, padx=16)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)
        self._build_sidebar(body)
        self._build_center(body)
        self._build_right(body)

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=PANEL, width=180)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        tk.Frame(parent, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        self._section_title(sidebar, "◈ SYSTEM STATUS")
        self.status_items = {}
        for label, color in [
            ("VOICE ENGINE",  GREEN), ("SPEECH RECOG.", GREEN),
            ("BERT MODEL",    GREEN), ("CAMERA",        RED),
            ("STOCK DATA",    GREEN), ("NASA API",      GREEN),
        ]:
            row = tk.Frame(sidebar, bg=PANEL)
            row.pack(fill=tk.X, padx=12, pady=3)
            dot = tk.Canvas(row, width=8, height=8, bg=PANEL, highlightthickness=0)
            dot.create_oval(1, 1, 7, 7, fill=color, outline="")
            dot.pack(side=tk.LEFT)
            tk.Label(row, text=f"  {label}", bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(side=tk.LEFT)
            self.status_items[label] = dot

        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=12, pady=10)
        self._section_title(sidebar, "◈ QUICK COMMANDS")

        for label, cmd in [
            ("📅  Date & Time",  "date"),
            ("📸  Screenshot",   "screenshot"),
            ("💻  CPU Usage",    "cpu"),
            ("🌌  NASA News",    "space news"),
            ("📈  Stock Tips",   "stock recommendation"),
            ("😄  Tell a Joke",  "tell me a joke"),
            ("🧠  My Emotion",   "how do i feel"),
            ("🗒️  Write a Note", "write a note"),
        ]:
            btn = tk.Button(
                sidebar, text=label, bg=PANEL, fg=TEXT, font=FONT_SMALL,
                bd=0, relief=tk.FLAT, activebackground=BORDER, activeforeground=ACCENT,
                cursor="hand2", anchor="w", padx=14, pady=4,
                command=lambda c=cmd: self._quick_cmd(c)
            )
            btn.pack(fill=tk.X, padx=8, pady=1)
            btn.bind("<Enter>", lambda e, b=btn: b.config(fg=ACCENT))
            btn.bind("<Leave>", lambda e, b=btn: b.config(fg=TEXT))

    def _build_center(self, parent):
        center = tk.Frame(parent, bg=BG)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        log_frame = tk.Frame(center, bg=PANEL)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self._section_title(log_frame, "◈ CONVERSATION LOG")

        self.log_area = scrolledtext.ScrolledText(
            log_frame, bg=BG, fg=TEXT, font=FONT_LOG,
            bd=0, relief=tk.FLAT, wrap=tk.WORD,
            state=tk.DISABLED, insertbackground=ACCENT, selectbackground=BORDER,
        )
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log_area.tag_config("JARVIS", foreground=ACCENT)
        self.log_area.tag_config("USER",   foreground=GREEN)
        self.log_area.tag_config("SYSTEM", foreground=TEXT_DIM)
        self.log_area.tag_config("ERROR",  foreground=RED)
        self.log_area.tag_config("ts",     foreground=TEXT_DIM)

        tk.Frame(center, bg=BORDER, height=1).pack(fill=tk.X)
        input_frame = tk.Frame(center, bg=PANEL, height=48)
        input_frame.pack(fill=tk.X)
        input_frame.pack_propagate(False)

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            input_frame, textvariable=self.input_var,
            bg=BG, fg=TEXT, insertbackground=ACCENT,
            font=FONT_MONO, bd=0, relief=tk.FLAT,
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=12)
        self.input_entry.bind("<Return>", self._on_text_submit)
        tk.Label(input_frame, text="▸ TYPE OR PRESS MIC",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(side=tk.LEFT)
        self.mic_btn = tk.Button(
            input_frame, text="🎤", bg=PANEL, fg=ACCENT,
            font=("Courier", 14), bd=0, relief=tk.FLAT,
            activebackground=BORDER, cursor="hand2",
            width=3, command=self._toggle_listen
        )
        self.mic_btn.pack(side=tk.RIGHT, padx=10)

    def _build_right(self, parent):
        tk.Frame(parent, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)
        right = tk.Frame(parent, bg=PANEL, width=180)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self._section_title(right, "◈ USER PROFILE")
        for key, val in [
            ("USER",    USERNAME),
            ("SYSTEM",  BOTNAME),
            ("OS",      OS),
            ("SESSION", datetime.now().strftime("%H:%M")),
        ]:
            row = tk.Frame(right, bg=PANEL)
            row.pack(fill=tk.X, padx=12, pady=3)
            tk.Label(row, text=f"{key}:", bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL, width=8, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=PANEL, fg=ACCENT, font=FONT_SMALL).pack(side=tk.LEFT)

        tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X, padx=12, pady=10)
        self._section_title(right, "◈ ACTIVITY")
        self.activity_log = tk.Listbox(
            right, bg=BG, fg=TEXT_DIM, font=FONT_SMALL,
            bd=0, relief=tk.FLAT, selectbackground=BORDER,
            highlightthickness=0, height=10,
        )
        self.activity_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _build_footer(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)
        footer = tk.Frame(self.root, bg=PANEL, height=28)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        self.status_label = tk.Label(footer, text="● STANDBY", bg=PANEL,
                                     fg=GREEN, font=FONT_SMALL)
        self.status_label.pack(side=tk.LEFT, padx=14, pady=6)
        tk.Label(footer, text="JARVIS AI  //  PYTHON VOICE ASSISTANT",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(side=tk.RIGHT, padx=14)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _section_title(self, parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL, anchor="w").pack(fill=tk.X, padx=12, pady=(10, 6))

    def log_message(self, sender: str, message: str, color: str = None):
        self.log_area.configure(state=tk.NORMAL)
        ts  = datetime.now().strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{ts}] ", "ts")
        tag = sender if sender in ("JARVIS", "USER", "SYSTEM", "ERROR") else "SYSTEM"
        if color:
            self.log_area.tag_config(f"c_{sender}", foreground=color)
            tag = f"c_{sender}"
        self.log_area.insert(tk.END, f"{sender}: ", tag)
        self.log_area.insert(tk.END, f"{message}\n")
        self.log_area.configure(state=tk.DISABLED)
        self.log_area.see(tk.END)
        self.activity_log.insert(tk.END, f"{ts} {sender[:6]}")
        self.activity_log.see(tk.END)

    def set_status(self, text: str, color: str = GREEN):
        self.status_label.config(text=f"● {text}", fg=color)

    def _start_clock(self):
        def tick():
            self.clock_label.config(text=datetime.now().strftime("%H:%M:%S  //  %d %b %Y"))
            self.root.after(1000, tick)
        tick()

    def _animate_status(self):
        self._blink_idx = 0
        def blink():
            if not self.listening:
                states = [("● STANDBY", GREEN), ("○ STANDBY", TEXT_DIM)]
                txt, col = states[self._blink_idx % 2]
                self.status_label.config(text=txt, fg=col)
                self._blink_idx += 1
            self.root.after(1200, blink)
        blink()

    # ── Interaction ────────────────────────────────────────────────────────────

    def _on_text_submit(self, event=None):
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self.log_message("USER", text, color=GREEN)

        # Multi-turn: waiting for memory content
        if self._pending_memory:
            self._pending_memory = False
            with open(os.path.join(BASE_DIR, 'data.txt'), 'w') as f:
                f.write(text)
            resp = f"I'll remember that: {text}"
            self.log_message("JARVIS", resp, color=ACCENT)
            speak_async(resp)
            return

        # Multi-turn: waiting for investment goals
        if self._pending_stock:
            self._pending_stock = False
            self.set_status("ANALYZING...", ACCENT)
            threading.Thread(target=self._stock_thread, args=(text,), daemon=True).start()
            return

        self._process_command(text)

    def _stock_thread(self, goals: str):
        try:
            stocks = generate_recommendations(goals)
            result = (f"Recommended stocks: {', '.join(set(stocks))}"
                      if stocks else "No suitable recommendations found.")
            self.root.after(0, lambda r=result: (
                self.log_message("JARVIS", r, color=ACCENT),
                speak_async(r)
            ))
        except Exception as e:
            self.root.after(0, lambda: self.log_message("ERROR", str(e), color=RED))
        finally:
            self.root.after(0, lambda: self.set_status("STANDBY"))

    def _toggle_listen(self):
        if self.listening:
            return
        self.listening = True
        self.mic_btn.config(fg=RED)
        self.set_status("LISTENING...", RED)
        self.log_message("SYSTEM", "Listening for voice input...", color=TEXT_DIM)
        threading.Thread(target=self._listen_thread, daemon=True).start()

    def _listen_thread(self):
        try:
            query = take_user_input()
            self.root.after(0, lambda: self._on_voice_result(query))
        except Exception as e:
            self.root.after(0, lambda: self.log_message("ERROR", str(e), color=RED))
        finally:
            self.root.after(0, self._reset_mic)

    def _on_voice_result(self, query: str):
        if query and query != 'None':
            self.log_message("USER", query, color=GREEN)
            self._process_command(query)

    def _reset_mic(self):
        self.listening = False
        self.mic_btn.config(fg=ACCENT)
        self.set_status("STANDBY")

    def _quick_cmd(self, cmd: str):
        self.log_message("USER", cmd, color=GREEN)
        self._process_command(cmd)

    def _process_command(self, query: str):
        self.set_status("PROCESSING...", ACCENT)
        threading.Thread(target=self._run_command, args=(query,), daemon=True).start()

    def _run_command(self, query: str):
        try:
            result = process_query(query, say=speak_async, joke_state=self.joke_state)

            if result == _EXIT:
                self.root.after(1500, self.root.destroy)
                return

            if result == _AWAIT_MEMORY:
                self._pending_memory = True
                self.root.after(0, lambda: self.log_message(
                    "JARVIS", "What should I remember? Type it below.", color=ACCENT))
                return

            if result == _AWAIT_STOCK:
                self._pending_stock = True
                self.root.after(0, lambda: self.log_message(
                    "JARVIS", "What are your investing goals? Type them below.", color=ACCENT))
                return

            if result:
                self.root.after(0, lambda r=result: self.log_message("JARVIS", r, color=ACCENT))

        except Exception as e:
            self.root.after(0, lambda: self.log_message("ERROR", str(e), color=RED))
        finally:
            self.root.after(0, lambda: self.set_status("STANDBY"))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if '--cli' in sys.argv:
        run_cli()
    else:
        root = tk.Tk()
        JarvisGUI(root)
        root.mainloop()