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

def predict(image_data, filename):
    image = Image.open(io.BytesIO(image_data))
     # Create the predictions directory if it doesn't exist
    save_dir = './predictions'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = yolo_model(image)
    print(results)
    for i, result in enumerate(results):
        prediction_path = os.path.join(save_dir, f'{filename}_prediction.png')
        result.save(prediction_path)
    
    return prediction_path

@app.route('/predict', methods=['POST'])
async def predict_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image detected. Please upload the file.'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            filename = os.path.splitext(file.filename)[0]  # Extract filename without extension
            # Use asyncio to asynchronously call the predict function
            prediction_path = await asyncio.to_thread(predict, file.read(), filename)
            return await asyncio.to_thread(send_file, prediction_path)
    except Exception as e:  # Catch all exceptions
        print(e)
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
