import os
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLOv8m model
yolo_model = YOLO('./brain_tumor_adamax.pt')

def predict(image_data):
    image = Image.open(io.BytesIO(image_data))
    results = yolo_model(image)
    save_dir = './predictions'
    os.makedirs(save_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        prediction_path = os.path.join(save_dir, f'prediction.png')
        result.save(prediction_path)
    
    return prediction_path

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No image detected. Please upload the file.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        prediction_path = predict(file.read())
        return send_file(prediction_path)

if __name__ == '__main__':
    app.run(debug=True)
