from flask import Flask, jsonify, request
from flask_cors import CORS
from google.cloud import storage
import tempfile
import uuid
import os

app = Flask(__name__)
CORS(app)  

BUCKET_NAME = "cityscapes-dataset-package3"

storage_client = storage.Client()

def upload_to_bucket(bucket_name, file_obj, destination_blob_name, content_type):
    bucket = storage_client.bucket(bucket_name)
    print("Bucket:", bucket)
    blob = bucket.blob(destination_blob_name)
    print("Blob:", blob)
    print("Content type:", content_type)
    blob.upload_from_file(file_obj)
    print("Uploaded from file!")
    # blob.make_public() 
    print("Blob URL:", blob.public_url)
    return blob.public_url

@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from Python on Cloud Run!"})

def remove_color_cast(input_image_path):
    return input_image_path

@app.route('/api/process', methods=['POST'])
def process_image():
    print("[INFO] Received request to process image")
    if 'image' not in request.files:
        print("[ERROR] No image part in request.files")
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == '':
        print("[ERROR] Empty filename")
        return jsonify({'error': 'No selected file'}), 400

    print("[INFO] Received file:", image.filename)

    # Get file extension
    filename_ext = os.path.splitext(image.filename)[1].lower()
    if filename_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
        return jsonify({'error': 'Unsupported file type'}), 400

    # Create a temporary file for the input image
    with tempfile.NamedTemporaryFile(suffix=filename_ext, delete=False) as temp_input:
        image.save(temp_input.name)
        input_path = temp_input.name
        print("[INFO] Saved the image into /tmp")
        
    try:
        # Generate a unique filename for storage
        filename = f"{uuid.uuid4()}_{image.filename}"
        print("[INFO] Made filename with uuid")
        
        # Reset the file pointer and upload original image
        image.seek(0)
        before_url = upload_to_bucket(
            BUCKET_NAME, 
            image, 
            f"uploads/before_{filename}", 
            content_type=image.content_type or 'image/jpeg'
        )
        print("[#] Successfully uploaded input image to bucket")

        # Process the image to remove color cast
        output_path = remove_color_cast(input_path)
        print("[#] Successfully processed image")
        
        # Upload the processed image
        with open(output_path, 'rb') as processed_file:
            after_url = upload_to_bucket(
                BUCKET_NAME,
                processed_file,
                f"uploads/after_{filename}",
                content_type=image.content_type or 'image/jpeg'
            )
        print("[#] Successfully uploaded output image to bucket")
        
        # Clean up temporary files
        os.unlink(input_path)
        if output_path != input_path:
            os.unlink(output_path)

        print("[#] Successfully cleaned up temporary files")
        
        return jsonify({
            'message': 'Image processed successfully',
            'before_url': before_url,
            'after_url': after_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
