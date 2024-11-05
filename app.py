from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
from tensorflow.keras.models import load_model
from flask_cors import CORS
from googleapiclient.discovery import build
import random
import wave
import speech_recognition as sr

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Load your models here
image_model = load_model('discriminator_mode.h5')
# Assuming you have a custom speech recognition model
speech_model = load_model('spee.h5')

api_key = 'AIzaSyDtBXA6N6kbHTsXJB4RMCBZHJ6KlmUWfhk'
youtube = build('youtube', 'v3', developerKey=api_key)

# Function to get emotion label for FER2013 dataset
def get_emotion_label(probabilities):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_label = emotions[np.argmax(probabilities)]
    return emotion_label

def search_youtube(query):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=5
    )
    response = request.execute()
    videos = []
    for item in response['items']:
        video = {
            'title': item['snippet']['title'],
            'video_id': item['id']['videoId'],
            'thumbnail': item['snippet']['thumbnails']['default']['url']
        }
        videos.append(video)
    return videos

# Function for image prediction
def predict_emotion(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((28, 28))  # Resize according to your model's requirement
    if image.mode != "L":
        image = image.convert("L")
    image_array = np.array(image) / 255.0  # Normalize the image array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    _, emotion_prob = image_model.predict(image_array)
    predicted_emotion = get_emotion_label(emotion_prob)
    return predicted_emotion

# Function for audio prediction using speech recognition
def predict_audio_emotion(audio_data):
    recognizer = sr.Recognizer()
    
    # Use a wave file to read audio data
    with wave.open(audio_data, 'rb') as source:
        audio = recognizer.record(source)  # Read the entire audio file

    try:
        # Recognize speech using your custom model or Google's Speech Recognition
        text = recognizer.recognize_google(audio)
        print("Recognized text:", text)
        # Here, you can analyze the recognized text to predict emotion or use a separate model
        # For example, you might have a function `analyze_text_for_emotion(text)` to determine emotion
        # predicted_emotion = analyze_text_for_emotion(text)
        return predicted_emotion  # Return the emotion based on text analysis
    except sr.UnknownValueError:
        return "Unable to recognize speech"
    except sr.RequestError as e:
        return "Could not request results from Google Speech Recognition service; {0}".format(e)

youtube_search_queries = { 
    "Angry": ["Technology", "Gaming", "Entertainment", "Music", "Vlogs and Lifestyle", 
              "Food and Cooking", "Animals and Pets", "Travel and Adventure", "Comedy", 
              "DIY and Crafts", "Motivational and Inspirational"],
    'Disgust': ['Disgust music', 'Disgust philosophy'],
    "Fear": ["Gaming (Horror Gaming)", "Entertainment", "Music", "Vlogs and Lifestyle", 
             "Food and Cooking", "Animals and Pets", "Travel and Adventure", "Comedy", 
             "DIY and Crafts", "Motivational and Inspirational"],
    'Happy': ['Happy music', 'Happy philosophy'], 
    "Sad": ["Gaming", "Entertainment", "Music", "Vlogs and Lifestyle", "Food and Cooking", 
            "Animals and Pets", "Travel and Adventure", "Comedy", "DIY and Crafts", 
            "Motivational and Inspirational"],
    'Surprise': ['Surprise music', 'Surprise philosophy'],
    'Neutral': ['philosophy',"Education", "Technology", "History", "Finance and Business"]
}

# Route to handle combined image and audio prediction
@app.route('/predict_combined', methods=['POST'])
def predict_combined():
    try:
        # Get image file from the request
        image_file = request.files.get('image')
        image_data = image_file.read() if image_file else None
        
        # Get audio file from the request
        audio_file = request.files.get('audio')
        audio_data = audio_file.read() if audio_file else None
        
        # Predict emotion from image if provided
        if image_data:
            predicted_image_emotion = predict_emotion(image_data)
        else:
            predicted_image_emotion = None

        # Predict emotion from audio if provided
        if audio_data:
            # Save audio to a temporary file
            with open('temp_audio.wav', 'wb') as f:
                f.write(audio_data)
            predicted_audio_emotion = predict_audio_emotion('temp_audio.wav')
        else:
            predicted_audio_emotion = None
        
        # Combine predictions
        if predicted_image_emotion and predicted_audio_emotion:
            combined_emotion = predicted_image_emotion  # You could implement logic to choose one
        elif predicted_image_emotion:
            combined_emotion = predicted_image_emotion
        elif predicted_audio_emotion:
            combined_emotion = predicted_audio_emotion
        else:
            return jsonify({'error': 'No valid image or audio file uploaded.'}), 400

        # Fetch related YouTube videos based on the combined emotion
        query = random.choice(youtube_search_queries[combined_emotion])
        print(query)
        videos = search_youtube(query)
        return jsonify({'prediction': combined_emotion, 'videos': videos}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
