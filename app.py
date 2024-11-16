from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import os
import tempfile

# Load the trained Keras model
model = tf.keras.models.load_model('cnn_lstm_workout_model.h5')

# Define the class labels
class_labels = [
    'hammer curl', 'barbell biceps curl', 'incline bench press', 'chest fly machine', 'hip thrust', 'lat pulldown', 'decline bench press', 'deadlift', 'bench press', 'lateral raise', 'pull Up', 'shoulder press', 'leg extension', 'push-up', 't bar row', 'plank', 'leg raises', 'romanian deadlift', 'squat' 'russian twist', 'tricep Pushdown'
]

app = Flask(__name__)

def preprocess_frame(frame):
    """Preprocess a single video frame to match the model's input shape."""
    img = Image.fromarray(frame)
    img = img.resize((64, 64))  # Resize to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img_array

def extract_frames(video_path, sequence_length=30):
    """Extract and preprocess frames from the video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_frame(frame)
        frames.append(preprocessed_frame)
    cap.release()
    
    # Pad with empty frames if video is too short
    if len(frames) < sequence_length:
        frames += [np.zeros((64, 64, 3))] * (sequence_length - len(frames))
    return np.array(frames)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)
        temp_file_path = temp_file.name

    # Extract and preprocess frames
    input_sequence = extract_frames(temp_file_path)
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(input_sequence)
    class_index = np.argmax(prediction, axis=1)[0]
    exercise_name = class_labels[class_index]

    # Clean up temp file
    os.remove(temp_file_path)
    
    return jsonify({'exercise_name': exercise_name})

if __name__ == '__main__':
    app.run(debug=True)
