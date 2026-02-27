import requests
import threading
import os
import time
from datetime import datetime, timedelta
import random
import knowledge
import torch 
from transformers import BertTokenizer, BertModel
import cv2
import numpy as np
from stock import generate_recommendations
import webbrowser
import pyjokes
from keyboard import press_and_release
from help import speak, cpu, SMALL_TALK_PHRASES, screenshot, startup, take_user_input, create_file
from help import open_calculator, open_camera, open_cmd, open_notepad
from help import play_on_youtube, search_on_google, search_on_wikipedia, find_my_ip
import tkinter as tk
from PIL import Image, ImageTk
from dates import add_special_day, check_special_days
from config import USERNAME, BOTNAME, NASA_API_KEY, BASE_DIR


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
            
class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

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
            aspect_ratio = float(w)/h if h != 0 else 0

            # Extract features from the contour
            features = np.array([[area, aspect_ratio]], dtype=np.float32)

            # Recognize the gesture using the trained model
            ret, results, neighbours, dist = self.gesture_model.findNearest(features, k=5)

            # Check the results
            if results[0] == 1:
                return "Thumbs up!"
            elif results[0] == 2:
                return "Waving!"
            elif results[0] == 3:
                return "Pointing!"

        return "Unknown gesture"



def chrome_automation(command):
    query = str(command)
    key_mappings = {
        'new tab': 'ctrl + t',
        'close tab': 'ctrl + w',
        'new window': 'ctrl + n',
        'history': 'ctrl + h',
        'download': 'ctrl + j'
    }
    for action, keys in key_mappings.items():
        if action in query:
            press_and_release(keys)
            return
    if 'switch tab' in query:
        tab_num = query.replace("switch tab ", "").replace("to", "").strip()
        press_and_release(f'ctrl + {tab_num}')        
        
    
def notepad():
    speak("Tell me the query. I am ready to write.")
    writes = take_user_input().lower()
    if writes == 'None':
        speak("I didn't catch that, please try again.")
        return
    timestamp = datetime.now().strftime("%H-%M")
    filename = f"{timestamp}-note.txt"
    folder_path = os.path.join(BASE_DIR, "Database", "Notepad")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(writes)
    
    speak(f"Note saved as {filename}")
    
def CloseNotepad():
    speak("Closing notepad.")
    os.system("taskkill /f /im notepad.exe")
    

def date():
    current_date = datetime.now().strftime("%d/%m/%Y")
    speak(f"The current date is {current_date}")
    print(f"The current date is {current_date}")
    
# Greet the user
def greet_user():
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        hour = datetime.now().hour
        greeting = "Good Morning" if 6 <= hour < 12 else "Good Afternoon" if 12 <= hour < 16 else "Good Evening"
        speak(f"{greeting} {USERNAME}. I am {BOTNAME}. How may I assist you?")
        
        if ret:
            face_detector = FaceDetector()
            faces = face_detector.detect_faces(frame)
            speak("I am able to see you!" if len(faces) > 0 else "I don't see anyone")
        cap.release()
    except Exception as e:
        print(f"Camera error: {e}")
        speak(f"{greeting} {USERNAME}. I am {BOTNAME}. How may I assist you?")


#Detect emotions
def detect_emotions():
    try:
        user_input = take_user_input()
        if user_input == 'None':
            return "I couldn't understand what you said."
            
        inputs = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        sentiment_scores = outputs.last_hidden_state[:, 0, :]
        max_index = torch.argmax(sentiment_scores)
        emotion_labels = ['happiness', 'sadness', 'neutral']
        emotion_index = max_index.item() if 0 <= max_index.item() < len(emotion_labels) else 2
        emotion_score = int(sentiment_scores[0][emotion_index].item())
        emotion = emotion_labels[emotion_index]
        return f"I sense that you're feeling {emotion} with a score of {emotion_score}."
    except Exception as e:
        print(f"Error detecting emotions: {e}")
        return "I'm having trouble analyzing your emotions right now."


def calculate():
    try:
        speak("What operation do you want to make?")
        operation = take_user_input().lower()
        if operation == 'None':
            speak("I didn't catch that.")
            return
            
        speak("Enter the first number")
        num1_str = take_user_input()
        if num1_str == 'None':
            speak("I didn't catch that.")
            return
        
        speak("Enter the second number")
        num2_str = take_user_input()
        if num2_str == 'None':
            speak("I didn't catch that.")
            return
        
        # Try to convert to float
        try:
            num1 = float(num1_str)
            num2 = float(num2_str)
        except ValueError:
            speak("Sorry, I need valid numbers.")
            return
        
        operations = {
            "addition": num1 + num2,
            "add": num1 + num2,
            "subtraction": num1 - num2,
            "subtract": num1 - num2,
            "multiplication": num1 * num2,
            "multiply": num1 * num2,
            "division": num1 / num2 if num2 != 0 else 'undefined'
        }
        result = operations.get(operation, "Operation not supported")
        speak(f"The result is {result}")
    except Exception as e:
        print(f"Calculation error: {e}")
        speak("Sorry, there was an error with the calculation.")
        
# Ollama integration (optional - requires Ollama to be running)
def talk_to_ollama(message):
    payload = {
        "model": "llama2",
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None

def talk_to_ollama_concurrently(messages):
    threads = []
    responses = []
    for message in messages:
        thread = threading.Thread(target=lambda msg: responses.append(talk_to_ollama(msg)), args=(message,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    return responses

def talk_to_user():
    speak("What do you want to talk about?")
    topic = take_user_input()
    if topic != 'None':
        messages = [topic]
        responses = talk_to_ollama_concurrently(messages)
        if responses and responses[0]:
            try:
                response_text = responses[0]['message']['content']
                speak(response_text)
                print(response_text)
            except (KeyError, TypeError):
                speak("I'm having trouble connecting to the conversation system.")
        else:
            speak("I'm having trouble connecting to the conversation system. Make sure Ollama is running.")
        
def nasa_news(date_param):
    try:
        url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}&date={date_param}"
        r = requests.get(url)
        data = r.json()
        
        info = data.get('explanation', 'No explanation available.')
        title = data.get('title', 'Astronomy Picture of the Day')
        image_url = data.get('url', '')
        
        if not image_url:
            speak("No image available today.")
            return
            
        image_r = requests.get(image_url)
        file_name = f"{date_param}.jpg"

        folder_path = os.path.join(BASE_DIR, "Database", "NASA", "Images")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'wb') as f:
            f.write(image_r.content)
        
        img = Image.open(file_path)
        img.show()
        
        speak(f"Title: {title}")
        speak(f"According to NASA: {info}")
    except Exception as e:
        print(f"Error fetching NASA news: {e}")
        speak("Sorry, I couldn't fetch the news from NASA.")
    
def summary(body):
    try:
        # Try to use existing images or skip
        folder_path = os.path.join(BASE_DIR, "Database", "NASA", "Images")
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            if images:
                random_image = random.choice(images)
                image_path = os.path.join(folder_path, random_image)
                img = Image.open(image_path)
                img.show()

        url = f"https://hubblesite.org/api/v3/glossary/{body}"
        r = requests.get(url)
        data = r.json()

        if data:
            summary_text = data.get('definition', 'No definition found.')
            speak(f"According to NASA: {summary_text}")
        else:
            speak("No data available, try again later!")
    except Exception as e:
        speak(f"Error fetching summary: {e}")
        speak("Sorry, I couldn't fetch the summary.")
    
def astro(start_date, end_date):
    try:
        url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={NASA_API_KEY}"
        r = requests.get(url)
        data = r.json()
        total_astro = data['element_count']
        neo = data['near_earth_objects']

        speak(f"Total asteroids between {start_date} and {end_date} is: {total_astro}")
        speak("Extracting data for these asteroids.")

        for body in neo.get(str(start_date), []):
            asteroid_id = body['id']
            asteroid_name = body['name']
            asteroid_magnitude = body['absolute_magnitude_h']
            print(asteroid_id, asteroid_name, asteroid_magnitude)
            speak(f"Asteroid {name} with id {id_val} has an absolute magnitude of {absolute}.")
    except Exception as e:
        print(f"Error fetching asteroid data: {e}")
        speak("Sorry, I couldn't fetch the asteroid data.")
     
def on_closing():
    try:
        hour = datetime.now().hour
        if hour >= 21 or hour < 6:
            speak("Good night sir, take care!")
        else:
            speak('Have a good day sir!')
    except Exception as e:
        print(f"Error closing window: {e}")


def main():
    greet_user()
    joke_told = False
    check_special_days()
    
    while True:
        query = take_user_input()
        if query is None:
            continue
        
        query = query.lower()
            
        # Process user input
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
            speak('What do you want to search on Wikipedia, sir?')
            search_query = take_user_input().lower()
            if search_query != 'None':
                results = search_on_wikipedia(search_query)
                speak(f"According to Wikipedia, {results}")
                speak("For your convenience, I am printing it on the screen sir.")
                print(results)
        elif 'youtube' in query:
            speak('What do you want to play on Youtube, sir?')
            video = take_user_input().lower()
            if video != 'None':
                play_on_youtube(video)
        elif 'search on google' in query:
            speak('What do you want to search on Google, sir?')
            search_query = take_user_input().lower()
            if search_query != 'None':
                search_on_google(search_query)          
        elif 'date' in query:
            date()
        elif 'talk' in query:
            talk_to_user()          
        elif 'speed of internet' in query or 'internet speed' in query:
            webbrowser.open("https://fast.com")       
        elif 'open github' in query:
            webbrowser.open("https://github.com")            
        elif 'shutdown' in query:
            speak("Shutting down the system. Goodbye!")
            os.system("shutdown /s /t 1")
        elif 'cpu' in query:
            cpu()
            continue
        elif 'screenshot' in query:
            speak("Taking screenshot sir")
            screenshot()     
        elif 'tell me a joke' in query or 'joke' in query:
            if not joke_told:
                joke = pyjokes.get_joke()
                print(joke)
                speak(joke)
                joke_told = True
            else:
                speak("You've already heard a joke, sir. Would you like to hear another one?")
                response = take_user_input()
                if response and 'yes' in response.lower():
                    joke_told = False
                elif response and 'no' in response.lower():
                    speak("Okay, moving on to the next command.")  
                    continue
        elif 'how old am i' in query or 'my age' in query:
            age = knowledge.get_person_age(USERNAME)
            if age:
                speak(f"You are {age} years old")
            else:
                speak("I don't have your age information.")
        elif "favorite movie" in query:
            movie = knowledge.get_favorite_movie(USERNAME)
            if movie:
                speak(f"Your favorite movie is {movie}")
            else:
                speak("I don't have your favorite movie information.")
        elif "where do i live" in query or "my location" in query:
            city = knowledge.get_person_location(USERNAME)
            if city:
                speak(f"You live in {city}")
            else:
                speak("I don't have your location information.")
        elif 'stock recommendation' in query:
            speak("What are your investing goals?")
            investment_query = take_user_input()
            if investment_query != 'None':
                speak("It may take a while to analyze the stocks")
                recommended_stocks = generate_recommendations(investment_query)
                if recommended_stocks:
                    speak(f'Recommended stocks: {", ".join(set(recommended_stocks))}')
                else:
                    speak("I couldn't find any suitable stock recommendations.")
        elif "favorite cuisine" in query:
            cuisine = knowledge.get_favorite_cuisines(USERNAME)
            if cuisine and len(cuisine) >= 2:
                speak(f"Your favorite cuisines are {cuisine[0]} and {cuisine[1]}.")
            elif cuisine:
                speak(f"Your favorite cuisine is {cuisine[0]}.")
            else:
                speak("I don't have your favorite cuisine information.")
        elif "favorite book" in query:
            book_info = knowledge.get_favorite_book(USERNAME)
            if book_info:
                speak(f"Your favorite book is {book_info[0]} by {book_info[1]} in the {book_info[2]} genre.")
            else:
                speak("I don't have your favorite book information.")
        elif "favorite food" in query:
            favorite_food = knowledge.get_favorite_food(USERNAME)
            if favorite_food:
                speak("Your favorite foods are " + ", ".join(favorite_food))
            else:
                speak("I'm sorry, I don't have any information about your favorite foods.")      
        elif "how do i feel" in query or "my emotion" in query:
            response = detect_emotions()
            speak(response)
        elif 'exit' in query or 'sleep' in query or 'goodbye' in query:
            on_closing()
            break
        elif 'time' in query:
            strTime = datetime.now().strftime("%H:%M:%S")   
            speak(f"Sir, the time is {strTime}")
        elif 'space news' in query or 'nasa news' in query:
            nasa_news(datetime.now().date())        
        elif 'summary' in query:
            query = query.replace("jarvis", "")
            query = query.replace("about", "")
            query = query.replace("summary", "").strip()
            if query:
                summary(query)
            else:
                speak("What would you like a summary about?")
        elif 'asteroid' in query:
            start_date = datetime.now().date() - timedelta(days=30)
            end_date = datetime.now().date()
            astro(start_date, end_date)       
        elif 'google automation' in query or 'chrome automation' in query:
            chrome_automation(query)
        elif 'write a note' in query:
            notepad()
        elif 'dismiss that note' in query or 'close notepad' in query:
            CloseNotepad()        
        elif 'remember that' in query:
            speak("What should I remember sir?")
            rememberMessage = take_user_input().lower()
            if rememberMessage != 'None':
                speak("You told me to remember that " + rememberMessage)
                with open(os.path.join(BASE_DIR, 'data.txt'), 'w') as remember:
                    remember.write(rememberMessage)
        elif 'do you remember' in query:
            try:
                with open(os.path.join(BASE_DIR, 'data.txt'), 'r') as remember:
                    content = remember.read()
                    speak("You told me to remember that " + content)
            except FileNotFoundError:
                speak("You haven't asked me to remember anything yet.")
        elif 'open stackoverflow' in query or 'stack overflow' in query:
            webbrowser.open('https://stackoverflow.com')    
        elif 'calculation' in query or 'calculate' in query:
            calculate()
        elif 'create file' in query:
            create_file()
        elif 'add special day' in query:
            add_special_day()
        else:
            speak("I'm not sure how to help with that. Could you try rephrasing?")



if __name__ == '__main__':
    startup()
    main()
       
