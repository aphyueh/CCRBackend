import cv2
from datetime import timedelta
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from google.cloud import storage
from google.cloud.storage.blob import Blob
import io
from model.inference import remove_color_cast
import numpy as np
import os
from PIL import Image
import tempfile
import uuid

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://ccrwebsite-1005035431569.asia-southeast1.run.app"}})  

BUCKET_NAME = "cityscapes-dataset-package3"
TEMP_DIR = "/tmp/adjusted"

storage_client = storage.Client()

@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from Python on Cloud Run!"})

def upload_to_bucket(bucket_name, file_obj, destination_blob_name, content_type):
    bucket = storage_client.bucket(bucket_name)
    print("Bucket:", bucket, flush=True)
    blob = bucket.blob(destination_blob_name)
    print("Blob:", blob, flush=True)
    print("Content type:", content_type, flush=True)
    blob.upload_from_file(file_obj)
    print("Uploaded from file!")
    blob.make_public() 
    print("Blob URL:", blob.public_url)
    return blob.public_url

def main_remove_color_cast(
    img_bytes: bytes
) -> str:
    # Run your model to get output bytes
    output_bytes = remove_color_cast(img_bytes)
    print(f"[server] Successfully removed colour cast! Got output_bytes of length {len(output_bytes)}")

    # Save the result to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_output:
        temp_output.write(output_bytes)
        print(f"[server] Saved output to {temp_output.name}")
        return temp_output.name

@app.route('/api/process', methods=['POST'])
def process_image():
    print("[INFO] Received request to process image", flush=True)
    if 'image' not in request.files:
        print("[ERROR] No image part in request.files", flush=True)
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == '':
        print("[ERROR] Empty filename", flush=True)
        return jsonify({'error': 'No selected file'}), 400

    print("[*] Received file:", image.filename, flush=True)

    # Get file extension
    filename_ext = os.path.splitext(image.filename)[1].lower()
    if filename_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
        return jsonify({'error': 'Unsupported file type'}), 400

    # Create a temporary file for the input image
    with tempfile.NamedTemporaryFile(suffix=filename_ext, delete=False) as temp_input:
        image.save(temp_input.name)
        input_path = temp_input.name
        print("[*] Saved the image into /tmp", flush=True)
        
    try:
        with open(input_path, 'rb') as f:
            img_bytes = f.read()

        # Process the image and get a temporary output path
        output_path = main_remove_color_cast(img_bytes)
        print("[*] Successfully processed image", flush=True)

        return send_file(
            output_path,
            mimetype="image/png",
            as_attachment=False,
            download_name=f"processed_{str(uuid.uuid4())[:6]}_{image.filename}"
        )

        # # Reset the file pointer and upload original image
        # image.seek(0)
        # print("[*] Reset file pointer", flush=True)
        # before_blob_name = f"uploads/before_{filename}"
        # before_url = upload_to_bucket(
        #     BUCKET_NAME, 
        #     image, 
        #     f"uploads/before_{filename}", 
        #     content_type=image.content_type or 'image/jpeg'
        # )
        # print("[*] Successfully uploaded input image to bucket", flush=True)

        # after_blob_name = f"uploads/after_{filename}"
        
        # # Upload the processed image
        # with open(output_path, 'rb') as processed_file:
        #     after_url = upload_to_bucket(
        #         BUCKET_NAME,
        #         processed_file,
        #         after_blob_name,
        #         content_type=image.content_type or 'image/jpeg'
        #     )
        # print("[#] Successfully uploaded output image to bucket", flush=True)
                
        # Clean up temporary files
        # os.unlink(input_path)
        # if output_path != input_path:
        #     os.unlink(output_path)

        # print("[#] Successfully cleaned up temporary files", flush=True)
        
        # return jsonify({
        #     # 'message': 'Image processed successfully',
        #     # 'before_url': before_url,
        #     'after_url': after_url
        # })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adjust', methods=['POST'])
def adjust_image():
    print("[INFO] Received request to adjust image", flush=True)

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Get adjustment parameters
    try:
        brightness = float(request.form.get('brightness', 0))
        contrast = float(request.form.get('contrast', 0))
        saturation = float(request.form.get('saturation', 0))
        temperature = float(request.form.get('temperature', 0))
    except ValueError:
        return jsonify({'error': 'Invalid parameters'}), 400

    img = Image.open(image.stream).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0

    # Brightness
    arr += brightness / 100.0

    # Contrast: scale around mid gray
    arr = (arr - 0.5) * (1 + contrast / 100.0) + 0.5

    # Saturation
    gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis]
    arr = arr + (arr - gray) * (saturation / 100.0)

    # Temperature: warm/cool shift (simple RGB adjustment)
    arr[..., 0] += temperature / 100.0  # Red
    arr[..., 2] -= temperature / 100.0  # Blue

    arr = np.clip(arr, 0, 1)
    out = (arr * 255).astype(np.uint8)
    out_img = Image.fromarray(out)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

@app.route('/api/histogram', methods=['POST'])
def histogram():
    file = request.files['image']
    image = Image.open(file).convert('RGB')
    np_img = np.array(image)

    histogram_data = {}
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([np_img], [i], None, [256], [0, 256]).flatten()
        histogram_data[color] = hist.tolist()

    return jsonify(histogram_data)


if __name__ == "__main__":
    app.run(debug=True)
