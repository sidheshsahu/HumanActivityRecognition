import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Initialize the Flask app
app = Flask(__name__)

# Load your pre-trained model
model = load_model('your_model.h5')

# Define the activities
activities = [
    'Reading a book',
    'Riding a bicycle',
    'Watching a sunrise or sunset',
    'Listening to music'
]

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        # Save the uploaded image to the uploads directory
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Predict and get the activity
        img_array = load_and_preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_activity = activities[predicted_class_index]

        return render_template('index.html', predicted_activity=predicted_activity)

    return 'No file uploaded', 400

if __name__ == '__main__':
    app.run(debug=True)
