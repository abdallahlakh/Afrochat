from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login,logout
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from django.views.decorators.csrf import csrf_exempt

import nltk
from django.contrib.staticfiles import finders
from django.contrib.auth.decorators import login_required
import json
import cv2
import mediapipe as mp
import numpy as np
import pickle
from django.http import StreamingHttpResponse
from django.http import JsonResponse
import openai
import os
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
assets_dir = "assets/"
# Labels dictionary
labels_dict = {
    33: 'A', 34: 'B', 9: 'L', 0: 'C', 1: 'D', 2: 'E', 3: 'F', 4: 'G', 5: 'H', 6: 'I', 7: 'J', 8: 'K', 10: 'M', 
    11: 'N', 12: 'O', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'U', 19: 'V', 20: 'W', 21: 'X', 22: 'Y', 23: 'Z', 
    24: '1', 25: '2', 26: '3', 27: '4', 28: '5', 29: '6', 30: '7', 31: '8', 32: '9'
}



last_detected_character = ''

def gen_frames():
    global last_detected_character
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "ENTER" TO GIVE OTHER CARACTERE! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == 13:  # For Windows, use 13 for the Enter key
          break

        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())

                    # Prepare data for prediction
                    data_aux = []
                    x_, y_, z_ = [], [], []
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)
                        z_.append(landmark.z)

                    center_x, center_y, center_z = np.mean(x_), np.mean(y_), np.mean(z_)
                    for landmark in hand_landmarks.landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                        data_aux.extend([x - min(x_), y - min(y_), z - min(z_), distance_from_center])

                    # Predict gesture
                    if data_aux:
                        prediction = model.predict([np.asarray(data_aux)])
                        print("%%%%%%%%%",prediction)
                        detected_character = labels_dict[int(prediction[0])]
                        print("********",detected_character)
                        last_detected_character = detected_character
                        cv2.putText(frame, last_detected_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

from django.http import JsonResponse

def save_character(request):
    global last_detected_character
    print("nduwnundwund")
    return JsonResponse({'character': last_detected_character})

def home_view(request):
	return render(request,'home.html')


def about_view(request):
	return render(request,'about.html')


def contact_view(request):
	return render(request,'contact.html')
	
@login_required(login_url="login")
def img_detection_view(request):
    if request.method == 'POST':
        with open('response_chars.json', 'r', encoding='utf-8') as file:
            subtitles = json.load(file)
            concatenated_string = ''.join(subtitles['response_chars'])
        # Convert list of valid_words to a single string
        text = concatenated_string
        words = word_tokenize(text)
        tagged = nltk.pos_tag(words)
        tense = {}
        tense["future"] = len([word for word in tagged if word[1] == "MD"])
        tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]])
        tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
        tense["present_continuous"] = len([word for word in tagged if word[1] == "VBG"])

        stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've", 'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have', 'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])

        lr = WordNetLemmatizer()
        filtered_text = []
        for w, p in zip(words, tagged):
            if w not in stop_words:
                if p[1] in ['VBG', 'VBD', 'VBZ', 'VBN', 'NN']:
                    filtered_text.append(lr.lemmatize(w, pos='v'))
                elif p[1] in ['JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
                    filtered_text.append(lr.lemmatize(w, pos='a'))
                else:
                    filtered_text.append(lr.lemmatize(w))

        words = filtered_text
        temp = []
        for w in words:
            if w == 'I':
                temp.append('Me')
            else:
                temp.append(w)
        words = temp
        probable_tense = max(tense, key=tense.get)

        if probable_tense == "past" and tense["past"] >= 1:
            temp = ["Before"] + words
            words = temp
        elif probable_tense == "future" and tense["future"] >= 1:
            if "Will" not in words:
                temp = ["Will"] + words
                words = temp
        elif probable_tense == "present" and tense["present_continuous"] >= 1:
            temp = ["Now"] + words
            words = temp

        filtered_text = []
        for w in words:
            print("www",w)
            path = w + ".mp4"
            f = finders.find(path)
            if not f:
                for c in w:
                    filtered_text.append(c)
            else:
                filtered_text.append(w)
        words = filtered_text

        return render(request, 'index.html', {'words': words, 'text': text})
    else:
        return render(request, 'index.html')




def signup_view(request):
	if request.method == 'POST':
		form = UserCreationForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request,user)
			# log the user in
			return redirect('animation')
	else:
		form = UserCreationForm()
	return render(request,'signup.html',{'form':form})



def login_view(request):
	if request.method == 'POST':
		form = AuthenticationForm(data=request.POST)
		if form.is_valid():
			#log in user
			user = form.get_user()
			login(request,user)
			if 'next' in request.POST:
				return redirect(request.POST.get('next'))
			else:
				return redirect('animation')
	else:
		form = AuthenticationForm()
	return render(request,'login.html',{'form':form})


def logout_view(request):
	logout(request)
	return redirect("home")



@csrf_exempt
def chat(request):
    if request.method == 'POST':
        user_input = request.POST.get('userInput')
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        
        response_message = response.choices[0].message['content'].strip()
        
        # Convert the response message into a list of characters
        response_chars = [char for char in response_message if char not in ['.', ',', '!', '?', '\'']]
        # Create a dictionary to store the list
        response_dict = {'response_chars': response_chars}
        
        # Write the dictionary to a JSON file
        with open('response_chars.json', 'w') as json_file:
            json.dump(response_dict, json_file)
        
        return JsonResponse({'response': response_message})