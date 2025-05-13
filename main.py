from flask import Flask, jsonify, request
from flask_cors import CORS
from google.cloud import storage
from google.cloud.storage.blob import Blob
from datetime import timedelta
from tensorflow.keras.models import load_model
import tempfile
import uuid
import os
from model.inference import remove_colour_cast

app = Flask(__name__)
CORS(app)  

BUCKET_NAME = "cityscapes-dataset-package3"
model = load_model("model/model.keras")

storage_client = storage.Client()

@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from Python on Cloud Run!"})

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # adjust based on your model input
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

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

def remove_color_cast_to_file(
    img_bytes: bytes,
    brightness: float = 100.0,
    noise: float = 0.0,
    contrast: float = 100.0
) -> str:
    # Run your model to get output bytes
    output_bytes = remove_colour_cast(
        img_bytes,
        brightness_pct=brightness,
        noise_pct=noise,
        contrast_pct=contrast,
    )
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

    print("[INFO] Received file:", image.filename, flush=True)

    # Get file extension
    filename_ext = os.path.splitext(image.filename)[1].lower()
    if filename_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
        return jsonify({'error': 'Unsupported file type'}), 400

    # Create a temporary file for the input image
    with tempfile.NamedTemporaryFile(suffix=filename_ext, delete=False) as temp_input:
        image.save(temp_input.name)
        input_path = temp_input.name
        print("[INFO] Saved the image into /tmp", flush=True)
        
    try:
        # Generate a unique filename for storage
        filename = f"{uuid.uuid4()}_{image.filename}"
        print("[INFO] Made filename with uuid", flush=True)
        
        # Reset the file pointer and upload original image
        image.seek(0)
        original_width, original_height = image.size
        before_blob_name = f"uploads/before_{filename}"
        before_url = upload_to_bucket(
            BUCKET_NAME, 
            image, 
            f"uploads/before_{filename}", 
            content_type=image.content_type or 'image/jpeg'
        )
        print("[#] Successfully uploaded input image to bucket", flush=True)

        with open(input_path, 'rb') as f:
            img_bytes = f.read()

        # Process the image and get a temporary output path
        output_path = remove_color_cast_to_file(
            img_bytes,
            brightness=100.0,   # or dynamic values if available
            noise=0.0,
            contrast=100.0
        )

        print("[#] Successfully processed image", flush=True)

        after_blob_name = f"uploads/after_{filename}"
        # Upload the processed image
        with open(output_path, 'rb') as processed_file:
            after_url = upload_to_bucket(
                BUCKET_NAME,
                processed_file,
                f"uploads/after_{filename}",
                content_type=image.content_type or 'image/jpeg'
            )
        print("[#] Successfully uploaded output image to bucket", flush=True)
        
        content_type = image.content_type or "image/jpeg"
        
        # Clean up temporary files
        os.unlink(input_path)
        if output_path != input_path:
            os.unlink(output_path)

        print("[#] Successfully cleaned up temporary files", flush=True)
        
        return jsonify({
            'message': 'Image processed successfully',
            'before_url': before_url,
            'after_url': after_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
