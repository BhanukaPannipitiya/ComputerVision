from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model (justification: Reuses CNN for real-time inference, outcome 3)
model = load_model('garbage_classifier.h5')
categories = ['organic', 'metal', 'plastic']  # From notebook

@app.route('/')
def index():
    return render_template('index.html')  # Serves the frontend

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Preprocess (same as notebook: resize/normalize, outcome 1)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict (justification: Evaluates image with lightweight CNN, outcome 4)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred))  # For display
    
    return jsonify({
        'category': categories[class_idx],
        'confidence': f"{confidence:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True)  # Run on