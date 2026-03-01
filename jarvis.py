import os
import platform
import random
import threading
import webbrowser
from datetime import datetime, timedelta

import cv2
import numpy as np
import pyjokes
import requests
import torch
from PIL import Image
from keyboard import press_and_release
from transformers import BertModel, BertTokenizer

import knowledge
from config import BASE_DIR, BOTNAME, NASA_API_KEY, USERNAME
from dates import add_special_day, check_special_days
from help import (SMALL_TALK_PHRASES, cpu, create_file, find_my_ip,
                  open_calculator, open_camera, open_cmd, open_notepad,
                  play_on_youtube, screenshot, search_on_google,
                  search_on_wikipedia, speak, startup, take_user_input)
from stock import generate_recommendations

# Platform 
OS = platform.system()

# uppress OpenCV camera probe errors
cv2.setLogLevel(3)

# BERT (emotion detection) 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
EMOTION_LABELS = ['happiness', 'sadness', 'neutral']

# Face / Gesture Detection 
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
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            features = np.array([[area, aspect_ratio]], dtype=np.float32)
            _, results, _, _ = self.gesture_model.findNearest(features, k=5)
            gesture_map = {1: "Thumbs up!", 2: "Waving!", 3: "Pointing!"}
            if results[0] in gesture_map:
                return gesture_map[results[0]]
        return "Unknown gesture"


# Greet 
def greet_user() -> None:
    hour = datetime.now().hour
    greeting = "Good Morning" if 6 <= hour < 12 else "Good Afternoon" if 12 <= hour < 16 else "Good Evening"
    speak(f"{greeting} {USERNAME}. I am {BOTNAME}. How may I assist you?")
    print(f"{greeting} {USERNAME}. I am {BOTNAME}. How may I assist you?")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                faces = FaceDetector().detect_faces(frame)
                speak("I am able to see you!" if len(faces) > 0 else "I don't see anyone")
            cap.release()
        else:
            print("No camera available, skipping face detection.")
    except Exception as e:
        print(f"Camera error: {e}")


# Emotion Detection 
def detect_emotions() -> str:
    try:
        user_input = take_user_input()
        if user_input == 'None':
            return "I couldn't understand what you said."
        inputs = tokenizer.encode_plus(
            user_input, add_special_tokens=True, max_length=512,
            return_attention_mask=True, return_tensors='pt', truncation=True
        )
        outputs = bert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        scores = outputs.last_hidden_state[:, 0, :]
        idx = torch.argmax(scores).item()
        emotion_index = idx if 0 <= idx < len(EMOTION_LABELS) else 2
        emotion = EMOTION_LABELS[emotion_index]
        score = int(scores[0][emotion_index].item())
        return f"I sense that you're feeling {emotion} with a score of {score}."
    except Exception as e:
        print(f"Error detecting emotions: {e}")
        return "I'm having trouble analyzing your emotions right now."


# Calculator 
def calculate() -> None:
    try:
        speak("What operation do you want to make?")
        operation = take_user_input()
        if not operation or operation == 'None':
            speak("I didn't catch that.")
            return

        speak("Enter the first number")
        num1_str = take_user_input()
        speak("Enter the second number")
        num2_str = take_user_input()

        try:
            num1, num2 = float(num1_str), float(num2_str)
        except (ValueError, TypeError):
            speak("Sorry, I need valid numbers.")
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
        speak(f"The result is {result}")
    except Exception as e:
        print(f"Calculation error: {e}")
        speak("Sorry, there was an error with the calculation.")


# Notepad 
def notepad() -> None:
    speak("Tell me the query. I am ready to write.")
    text = take_user_input()
    if not text or text == 'None':
        speak("I didn't catch that, please try again.")
        return
    folder = os.path.join(BASE_DIR, "Database", "Notepad")
    os.makedirs(folder, exist_ok=True)
    filename = f"{datetime.now().strftime('%H-%M')}-note.txt"
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        f.write(text)
    speak(f"Note saved as {filename}")


def close_notepad() -> None:
    speak("Closing notepad.")
    if OS == 'Windows':
        os.system('taskkill /f /im notepad.exe')
    elif OS == 'Darwin':
        os.system('pkill -x TextEdit')
    else:
        os.system('pkill gedit')


# Chrome Automation 
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


# Ollama (optional) 
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


def talk_to_user() -> None:
    speak("What do you want to talk about?")
    topic = take_user_input()
    if topic and topic != 'None':
        responses = []
        t = threading.Thread(target=lambda: responses.append(_talk_to_ollama(topic)))
        t.start()
        t.join()
        if responses and responses[0]:
            try:
                speak(responses[0]['message']['content'])
            except (KeyError, TypeError):
                speak("I'm having trouble connecting to the conversation system.")
        else:
            speak("Make sure Ollama is running.")


# NASA 
def nasa_news(date_param) -> None:
    try:
        data = requests.get(
            f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}&date={date_param}"
        ).json()
        title     = data.get('title', 'Astronomy Picture of the Day')
        info      = data.get('explanation', 'No explanation available.')
        image_url = data.get('url', '')

        if not image_url:
            speak("No image available today.")
            return

        folder = os.path.join(BASE_DIR, "Database", "NASA", "Images")
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"{date_param}.jpg")

        with open(file_path, 'wb') as f:
            f.write(requests.get(image_url).content)

        Image.open(file_path).show()
        speak(f"Title: {title}")
        speak(f"According to NASA: {info}")
    except Exception as e:
        print(f"NASA error: {e}")
        speak("Sorry, I couldn't fetch the news from NASA.")


def summary(body: str) -> None:
    try:
        folder = os.path.join(BASE_DIR, "Database", "NASA", "Images")
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            if images:
                Image.open(os.path.join(folder, random.choice(images))).show()

        data = requests.get(f"https://hubblesite.org/api/v3/glossary/{body}").json()
        speak(f"According to NASA: {data.get('definition', 'No definition found.')}" if data else "No data available.")
    except Exception as e:
        speak(f"Sorry, I couldn't fetch the summary.")
        print(f"Summary error: {e}")


def astro(start_date, end_date) -> None:
    try:
        data = requests.get(
            f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={NASA_API_KEY}"
        ).json()
        speak(f"Total asteroids between {start_date} and {end_date} is: {data['element_count']}")
        speak("Extracting data for these asteroids.")
        for body in data['near_earth_objects'].get(str(start_date), []):
            speak(f"Asteroid {body['name']} with id {body['id']} has magnitude {body['absolute_magnitude_h']}.")
    except Exception as e:
        print(f"Asteroid error: {e}")
        speak("Sorry, I couldn't fetch the asteroid data.")


# Shutdown 
def on_closing() -> None:
    hour = datetime.now().hour
    speak("Good night sir, take care!" if hour >= 21 or hour < 6 else "Have a good day sir!")


# Main Loop 
def main() -> None:
    greet_user()
    check_special_days()
    joke_told = False

    while True:
        query = take_user_input()
        if not query:
            continue
        query = query.lower()

        if 'open notepad' in query:
            open_notepad()
        elif 'open command prompt' in query or 'open cmd' in query:
            open_cmd()
        elif 'open camera' in query:
            open_camera()
        elif 'open calculator' in query:
            open_calculator()
        elif 'ip address' in query:
            find_my_ip()
        elif 'wikipedia' in query:
            speak('What do you want to search on Wikipedia?')
            q = take_user_input()
            if q and q != 'None':
                result = search_on_wikipedia(q)
                speak(f"According to Wikipedia, {result}")
                print(result)
        elif 'youtube' in query:
            speak('What do you want to play on Youtube?')
            video = take_user_input()
            if video and video != 'None':
                play_on_youtube(video)
        elif 'search on google' in query:
            speak('What do you want to search on Google?')
            q = take_user_input()
            if q and q != 'None':
                search_on_google(q)
        elif 'date' in query:
            speak(f"The current date is {datetime.now().strftime('%d/%m/%Y')}")
        elif 'time' in query:
            speak(f"Sir, the time is {datetime.now().strftime('%H:%M:%S')}")
        elif 'talk' in query:
            talk_to_user()
        elif 'speed of internet' in query or 'internet speed' in query:
            webbrowser.open("https://fast.com")
        elif 'open github' in query:
            webbrowser.open("https://github.com")
        elif 'open stackoverflow' in query or 'stack overflow' in query:
            webbrowser.open('https://stackoverflow.com')
        elif 'shutdown' in query:
            speak("Shutting down the system. Goodbye!")
            if OS == 'Windows':
                os.system("shutdown /s /t 1")
            elif OS == 'Darwin':
                os.system("sudo shutdown -h now")
            else:
                os.system("shutdown -h now")
        elif 'cpu' in query:
            cpu()
        elif 'screenshot' in query:
            speak("Taking screenshot sir")
            screenshot()
        elif 'joke' in query or 'tell me a joke' in query:
            if not joke_told:
                joke = pyjokes.get_joke()
                print(joke)
                speak(joke)
                joke_told = True
            else:
                speak("You've already heard a joke. Would you like to hear another one?")
                response = take_user_input()
                if response and 'yes' in response:
                    joke_told = False
                elif response and 'no' in response:
                    speak("Okay, moving on.")
        elif 'how old am i' in query or 'my age' in query:
            age = knowledge.get_person_age(USERNAME)
            speak(f"You are {age} years old" if age else "I don't have your age information.")
        elif 'favorite movie' in query:
            movie = knowledge.get_favorite_movie(USERNAME)
            speak(f"Your favorite movie is {movie}" if movie else "I don't have your favorite movie information.")
        elif 'where do i live' in query or 'my location' in query:
            city = knowledge.get_person_location(USERNAME)
            speak(f"You live in {city}" if city else "I don't have your location information.")
        elif 'stock recommendation' in query:
            speak("What are your investing goals?")
            q = take_user_input()
            if q and q != 'None':
                speak("Analyzing the stocks, please wait.")
                stocks = generate_recommendations(q)
                speak(f'Recommended stocks: {", ".join(set(stocks))}' if stocks else "I couldn't find any suitable recommendations.")
        elif 'favorite cuisine' in query:
            cuisines = knowledge.get_favorite_cuisines(USERNAME)
            if cuisines and len(cuisines) >= 2:
                speak(f"Your favorite cuisines are {cuisines[0]} and {cuisines[1]}.")
            elif cuisines:
                speak(f"Your favorite cuisine is {cuisines[0]}.")
            else:
                speak("I don't have your favorite cuisine information.")
        elif 'favorite book' in query:
            book = knowledge.get_favorite_book(USERNAME)
            speak(f"Your favorite book is {book[0]} by {book[1]} in the {book[2]} genre." if book else "I don't have your favorite book information.")
        elif 'favorite food' in query:
            foods = knowledge.get_favorite_food(USERNAME)
            speak("Your favorite foods are " + ", ".join(foods) if foods else "I don't have your favorite food information.")
        elif 'how do i feel' in query or 'my emotion' in query:
            speak(detect_emotions())
        elif 'space news' in query or 'nasa news' in query:
            nasa_news(datetime.now().date())
        elif 'summary' in query:
            body = query.replace("jarvis", "").replace("about", "").replace("summary", "").strip()
            summary(body) if body else speak("What would you like a summary about?")
        elif 'asteroid' in query:
            astro(datetime.now().date() - timedelta(days=30), datetime.now().date())
        elif 'google automation' in query or 'chrome automation' in query:
            chrome_automation(query)
        elif 'write a note' in query:
            notepad()
        elif 'dismiss that note' in query or 'close notepad' in query:
            close_notepad()
        elif 'remember that' in query:
            speak("What should I remember sir?")
            msg = take_user_input()
            if msg and msg != 'None':
                with open(os.path.join(BASE_DIR, 'data.txt'), 'w') as f:
                    f.write(msg)
                speak(f"You told me to remember that {msg}")
        elif 'do you remember' in query:
            try:
                with open(os.path.join(BASE_DIR, 'data.txt'), 'r') as f:
                    speak("You told me to remember that " + f.read())
            except FileNotFoundError:
                speak("You haven't asked me to remember anything yet.")
        elif 'calculation' in query or 'calculate' in query:
            calculate()
        elif 'create file' in query:
            create_file()
        elif 'add special day' in query:
            add_special_day()
        elif 'exit' in query or 'sleep' in query or 'goodbye' in query:
            on_closing()
            break
        else:
            speak("I'm not sure how to help with that. Could you try rephrasing?")


if __name__ == '__main__':
    startup()
    main()