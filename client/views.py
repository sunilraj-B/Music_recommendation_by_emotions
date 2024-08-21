import http.client
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.shortcuts import redirect
from django.http import JsonResponse
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model # type: ignore
from collections import Counter
from tensorflow.keras.optimizers import Adam # type: ignore
from django.views.decorators.csrf import csrf_exempt
 
flag=0
def index(request):
    return render(request, 'index.html')
@csrf_exempt
def start_capturing(request):
    if request.method == 'POST' and request.POST.get('start_capturing') == 'true':
        # Start the capturing process here
        global flag 
        flag=1
        return redirect('detect_emotion')  # Redirect to the detect_emotion view
    else:
        # Handle invalid requests or direct access to this view
        return JsonResponse({'error': 'Invalid request'}, status=400)
def playlist_songs_by_emotion(request, emotion):
    # Define mapping of emotions to corresponding search terms for iTunes API
    emotion_mapping = {
        'sad': 'happy',
        'happy': 'sad',
        'angry': 'cool',
        'neutral': 'chill',
        'surprise': 'surprise',
        'normal': 'normal',
        'fear': 'fear',
        'disgust': 'disgust',
        'calm': 'calm',
        'none': 'weekend',
        'tamil':'love'
        # Add more emotions and corresponding search terms as needed
    }

    # Convert the emotion to lowercase to match the keys in emotion_mapping
    emotion = emotion.lower()

    if emotion in emotion_mapping:
        search_term = emotion_mapping[emotion]

        # Make a request to the iTunes Search API
        conn = http.client.HTTPSConnection("itunes.apple.com")
        conn.request("GET", f"/search?term='popular'+{search_term}&entity=musicTrack&limit=15")

        res = conn.getresponse()
        data = res.read().decode("utf-8")

        if res.status == 200:
            # Parse the JSON response
            response_data = json.loads(data)

            # Extract relevant track information from the response
            songs = []
            for result in response_data['results']:
                song = {
                    'track_name': result['trackName'],
                    'artist_name': result['artistName'],
                    'album_name': result['collectionName'],
                    'preview_url': result['previewUrl']
                }
                songs.append(song)

            # return JsonResponse({'songs': songs})
                context = {'songs': songs}
            return render(request, 'playlist.html', context)
        else:
            return JsonResponse({'error': 'Failed to fetch data from iTunes API'}, status=res.status)
    else:
        return JsonResponse({'error': f"No search term found for '{emotion}'."}, status=404)

def detect_emotion(request):
    global flag
    if flag==0:
        flag=1
        return render(request, 'capturing.html')
    # Load the emotion detection model
    model_path = r'D:\sunil projects\emotion_detection copy\emotion_detection_model_50epoch.h5'
    model = load_model(model_path)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Function to preprocess a single frame from the webcam
    def preprocess_frame(frame):
        resized_frame = cv2.resize(frame, (48, 48))
        # rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        # normalized_frame = rgb_frame / 255.0
        # preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = gray_frame / 255.0
        preprocessed_frame = np.expand_dims(normalized_frame, axis=-1)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

        return preprocessed_frame

    # Function to detect emotion from webcam feed for a specified duration (in seconds)
    def detect_emotion_from_webcam(duration=15):
        cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, adjust as necessary
        detected_emotions = ['none']
        
        start_time = time.time()  # Get the start time

        while (time.time() - start_time) < duration:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame from the webcam.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                preprocessed_face = preprocess_frame(face_region)

                # Predict emotion using the loaded model
                predictions = model.predict(preprocessed_face)
                predicted_class_index = np.argmax(predictions)
                predicted_emotion = emotion_labels[predicted_class_index]

                detected_emotions.append(predicted_emotion)

            # Convert detected emotions to a set to remove duplicates
            unique_emotions = list(set(detected_emotions))
           

            # Sleep for a short duration to prevent excessive CPU usage
            time.sleep(0.1)

        cap.release()
        cv2.destroyAllWindows()

        # Determine the most frequent emotion
        emotion_counts = Counter(detected_emotions)
        most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)
        # return most_frequent_emotion

    # Call the function to detect emotion from webcam feed
        # detected_emotion = detect_emotion_from_webcam()
        global flag 
        flag=0
     # Redirect to the playlist_songs_by_emotion view with the detected emotion
        return redirect('playlist_songs_by_emotion', emotion=most_frequent_emotion)
    

    # Call the function to detect emotion from webcam feed
    return detect_emotion_from_webcam()
