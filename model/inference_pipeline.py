import cv2
import io
import numpy as np
import os
from PIL import Image
import sys
from skimage import exposure, color as skimage_color, filters, restoration
import tensorflow as tf
from typing import Tuple


# Define if ximgproc is available (optional, for advanced dehazing)
try:
    import cv2.ximgproc
    CV2_XIMGPROC_AVAILABLE = True
    print("[*] cv2.ximgproc module found. Advanced dehazing option available.")
except ImportError:
    CV2_XIMGPROC_AVAILABLE = False
    print("[!] cv2.ximgproc module not found. Advanced dehazing (darkChannelDehazing) will be skipped.")


from .model import ColorCastRemoval
from .utils import tf_rgb_to_lab_normalized, tf_lab_normalized_to_rgb
from .input import IMG_WIDTH, IMG_HEIGHT

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ccrmodel.keras")
_model = None

EXTENSION_TO_FORMAT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".tiff": "TIFF",
    ".webp": "WEBP"
}

# --- Initialize Model ---
def initialize_model():
    global model
    global _model  
    if _model is not None:
        print("[INFO] Model already loaded. Skipping reload.", flush=True)
        model = _model
        return
    try:
        model = _get_model()
        print("[INFO] Model loaded successfully", flush=True)
    except Exception as e:
        print(f"[!] Model failed to load at startup: {e}", flush=True)

# --- Load Model ---
def _get_model() -> tf.keras.Model:
    global _model
    print(f"[*] Looking for model from {_MODEL_PATH}", flush=True)
    print(f"[*] Model file exists? {os.path.exists(_MODEL_PATH)}", flush=True)
    print(f"[*] Current directory: {os.getcwd()}", flush=True)
    print(f"[*] Model directory contents: {os.listdir(os.path.dirname(_MODEL_PATH))}", flush=True)
    
    if _model is None:
        if os.path.exists(_MODEL_PATH):
            # 1) Load the full .keras model
            _model = tf.keras.models.load_model(
                _MODEL_PATH,
                custom_objects={"ColorCastRemoval": ColorCastRemoval}
            )
            print(f"[*] Loaded Keras model from {_MODEL_PATH}", flush=True)
        else:
            print(f"[!] No model found in directory {_MODEL_PATH}", flush=True)
    return _model

# --- Pre-processing Function ---
def preprocess_image_hybrid(rgb_image_np_0_1):
    if rgb_image_np_0_1.ndim == 2: # Grayscale
        print("Warning: Input image is grayscale. Converting to RGB.", flush=True)
        rgb_image_np_0_1 = skimage_color.gray2rgb(rgb_image_np_0_1)
    elif rgb_image_np_0_1.shape[-1] == 4: # RGBA
        print("Warning: Input image is RGBA. Converting to RGB.", flush=True)
        rgb_image_np_0_1 = skimage_color.rgba2rgb(rgb_image_np_0_1)

    image_to_process = np.clip(rgb_image_np_0_1, 0.0, 1.0).astype(np.float32)
    img_uint8 = (image_to_process * 255).astype(np.uint8) # For OpenCV

    # 1. CLAHE
    # print("    Applying CLAHE...")
    lab_cv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_cv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Tune clipLimit
    cl = clahe.apply(l_channel)
    merged_lab_cv = cv2.merge((cl, a_channel, b_channel))
    processed_rgb_uint8 = cv2.cvtColor(merged_lab_cv, cv2.COLOR_LAB2RGB)
    processed_rgb_np = (processed_rgb_uint8 / 255.0).astype(np.float32)

    # 2. Gamma Correction (for brightness/haze perception)
    # print("    Applying Gamma Correction...")
    processed_rgb_np = exposure.adjust_gamma(processed_rgb_np, gamma=0.9) # Tune gamma (0.7-1.0)

    # --- More Advanced Dehazing (Optional - requires opencv-contrib) ---
    if CV2_XIMGPROC_AVAILABLE:
        try:
            # print("    Applying Advanced Dehazing (Dark Channel Prior)...")
            img_to_dehaze_uint8 = (processed_rgb_np * 255).astype(np.uint8)
            # Note: darkChannelDehazing expects BGR input
            img_bgr_to_dehaze = cv2.cvtColor(img_to_dehaze_uint8, cv2.COLOR_RGB2BGR)
            # Parameters for darkChannelDehazing might need tuning
            dehazed_bgr = cv2.ximgproc.darkChannelDehazing(img_bgr_to_dehaze, patchSize=15, omega=0.95, t0=0.1, ksize=-1)
            processed_rgb_uint8_dehazed = cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2RGB)
            processed_rgb_np = (processed_rgb_uint8_dehazed / 255.0).astype(np.float32)
        except Exception as e_dehaze:
            print(f"Warning: Advanced dehazing failed: {e_dehaze}. Skipping.", flush=True)


    return np.clip(processed_rgb_np, 0.0, 1.0).astype(np.float32)

# --- Post-processing Function ---
def postprocess_image_hybrid(rgb_image_np_0_1):
    image_to_process = np.clip(rgb_image_np_0_1, 0.0, 1.0).astype(np.float32)
    # 1. Sharpening
    # print("    Applying Sharpening...")
    sharpened_rgb_np = filters.unsharp_mask(image_to_process, radius=1.5, amount=1.2, channel_axis=-1, preserve_range=False)
    sharpened_rgb_np = np.clip(sharpened_rgb_np, 0.0, 1.0)

    # 2. Global Contrast Stretch (per-channel)
    # print("    Applying Contrast Stretch...")
    contrast_stretched_channels = []
    if sharpened_rgb_np.ndim == 3 and sharpened_rgb_np.shape[-1] == 3: # RGB image
        for i_ch in range(sharpened_rgb_np.shape[-1]):
            channel = sharpened_rgb_np[..., i_ch]
            p_low, p_high = np.percentile(channel, (1, 99))
            
            if p_low >= p_high:
                ch_min_val = channel.min()
                ch_max_val = channel.max()
                if ch_min_val < ch_max_val: p_low, p_high = ch_min_val, ch_max_val
                else: p_low, p_high = 0.0, 1.0 # Fallback for truly flat channel
            
            try:
                stretched_channel = exposure.rescale_intensity(channel, in_range=(p_low, p_high))
                contrast_stretched_channels.append(stretched_channel)
            except ValueError as e:
                print(f"Warning: rescale_intensity failed for channel {i_ch} ({e}). Using channel as is.", flush=True)
                contrast_stretched_channels.append(channel)
        
        if len(contrast_stretched_channels) == 3:
            final_rgb_np = np.stack(contrast_stretched_channels, axis=-1)
        else:
            print("Warning: Channel processing for contrast stretch failed. Using sharpened image.", flush=True)
            final_rgb_np = sharpened_rgb_np
    elif sharpened_rgb_np.ndim == 2: # Grayscale
        p_low, p_high = np.percentile(sharpened_rgb_np, (1, 99))
        if p_low >= p_high:
            img_min_val, img_max_val = sharpened_rgb_np.min(), sharpened_rgb_np.max()
            if img_min_val < img_max_val: p_low, p_high = img_min_val, img_max_val
            else: p_low, p_high = 0.0, 1.0
        try:
            final_rgb_np = exposure.rescale_intensity(sharpened_rgb_np, in_range=(p_low, p_high))
        except ValueError as e:
            print(f"Warning: rescale_intensity failed for grayscale image ({e}). Using sharpened image.",flush=True)
            final_rgb_np = sharpened_rgb_np
    else: # Unexpected shape
        print("Warning: Image is not RGB or Grayscale after sharpening. Skipping contrast stretch.", flush=True)
        final_rgb_np = sharpened_rgb_np

    # 3. Optional: Saturation Boost (after contrast and sharpening)
    print("    Applying Saturation Boost...", flush=True)
    hsv_img = skimage_color.rgb2hsv(final_rgb_np)
    hsv_img[..., 1] = np.clip(hsv_img[..., 1] * 1.1, 0, 1.0) # Tune factor
    final_rgb_np = skimage_color.hsv2rgb(hsv_img)

    return np.clip(final_rgb_np, 0.0, 1.0).astype(np.float32)

# --- Main Inference Function ---
def remove_color_cast(img_path: str) -> Tuple[bytes, str]:
    global _model  
    if _model is None:
        initialize_model()
        
    print(f"--- Starting Hybrid Inference ---", flush=True)
    print(f"\nProcessing image {img_path}", flush=True)
    
    try:
        img_bgr_uint8 = cv2.imread(img_path)
        img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
        raw_input_rgb_np = (img_rgb_uint8 / 255.0).astype(np.float32)
        # resized_input_rgb_np = cv2.resize(raw_input_rgb_np, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        print("  Applying pre-processing...", flush=True)
        preprocessed_rgb_np = preprocess_image_hybrid(raw_input_rgb_np)
        
        preprocessed_rgb_tensor_batched = tf.convert_to_tensor(np.expand_dims(preprocessed_rgb_np, axis=0), dtype=tf.float32)
        input_lab_tf_batched = tf_rgb_to_lab_normalized(preprocessed_rgb_tensor_batched)

        print("  Running CNN inference...", flush=True)
        enhanced_lab_tf_batched, _ = model(input_lab_tf_batched, training=False)
        enhanced_rgb_tf_batched = tf_lab_normalized_to_rgb(enhanced_lab_tf_batched)
        cnn_output_rgb_np = enhanced_rgb_tf_batched.numpy()[0]
        cnn_output_rgb_np = np.clip(cnn_output_rgb_np, 0.0, 1.0)

        print("  Applying post-processing...", flush=True)
        final_output_rgb_np = postprocess_image_hybrid(cnn_output_rgb_np)
        out_img = (final_output_rgb_np * 255).astype(np.uint8)

        # Create a PIL image
        pil_out = Image.fromarray(out_img)

        # Save to BytesIO buffer
        _, ext = os.path.splitext(os.path.basename(img_path))
        ext = ext.lower()
        img_format = EXTENSION_TO_FORMAT.get(ext, "PNG")  # Default to PNG if unknown

        buf = io.BytesIO()
        pil_out.save(buf, format=img_format)
        buf.seek(0)

        # Return image bytes
        image_bytes = buf.getvalue()
        return image_bytes, img_format

    except Exception as e:
        print(f"Error processing image {img_path}: {e}", flush=True)

    print("--- Hybrid Inference Finished ---", flush=True)