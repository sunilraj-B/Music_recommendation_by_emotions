# emotion_detection.py

import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model #type:ignore
from collections import Counter
from tensorflow.keras.optimizers import Adam #type:ignore

# Load the saved emotion detection model
model_path = r'C:\Users\sunil\OneDrive\Documents\sunil projects\emotion_detection\emotion_detection_model1.h5'
model = load_model(model_path)

# Assuming 'model' is the loaded model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess a single frame from the webcam
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (48, 48))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame / 255.0
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    return preprocessed_frame

# Function to detect emotion from webcam feed for a specified duration (in seconds)
def detect_emotion_from_webcam(duration=30):
    cap = cv2.VideoCapture(1)
    detected_emotions = []
    
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

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Emotion: {predicted_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            detected_emotions.append(predicted_emotion)

        cv2.imshow('Webcam - Emotion Detection', frame)

        # Check for key press to exit (waitKey returns ASCII value of key)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting the webcam feed...")
            break

    cap.release()
    cv2.destroyAllWindows()

    return detected_emotions

# Function to determine the most frequent emotion from the detected emotions list
def find_most_frequent_emotion(detected_emotions):
    # Count the occurrences of each emotion
    emotion_counts = Counter(detected_emotions)

    # Neutral emotions count is considered as half of the total detected emotions
    total_emotions = len(detected_emotions)
    neutral_count = total_emotions // 2

    # Subtract neutral_count from the count of Neutral if it exceeds neutral_count
    if 'Neutral' in emotion_counts:
        emotion_counts['Neutral'] = max(emotion_counts['Neutral'] - neutral_count, 0)

    # Find the emotion with the maximum count
    most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)
    max_count = emotion_counts[most_frequent_emotion]

    return most_frequent_emotion, max_count

# Function to suggest a playlist based on the most frequent emotion
def suggest_playlist_from_emotion(most_frequent_emotion):
    emotion_playlist_mapping = {
        'Angry': 'Angry Playlist',
        'Disgust': 'Disgust Playlist',
        'Fear': 'Fear Playlist',
        'Happy': 'Happy Playlist',
        'Neutral': 'Neutral Playlist',
        'Sad': 'Sad Playlist',
        'Surprise': 'Surprise Playlist'
    }

    if most_frequent_emotion in emotion_playlist_mapping:
        return emotion_playlist_mapping[most_frequent_emotion]
    else:
        return "Generic Playlist"  # Default playlist suggestion if emotion is not mapped

# Specify the duration (in seconds) for webcam feed and emotion detection
duration_seconds = 30
detected_emotions = detect_emotion_from_webcam(duration=duration_seconds)

# Find the most frequent emotion and its count
most_frequent_emotion, max_count = find_most_frequent_emotion(detected_emotions)

# Suggest a playlist based on the most frequent emotion
playlist_suggestion = suggest_playlist_from_emotion(most_frequent_emotion)

# Print the results
print(f"Most Frequent Emotion: {most_frequent_emotion} ({max_count} times)")
print(f"Suggested Playlist: {playlist_suggestion}")
