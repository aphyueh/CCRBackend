# model/inference.py
import cv2
import io
import numpy as np
import os
from PIL import Image, ImageEnhance
import tensorflow as tf

from .model import ColorCastRemoval


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")
_model = None

# Minimal LAB conversion utilities (extracted from utils.py)
def rgb_to_lab_normalized(rgb):
    """Convert RGB image to normalized LAB space (L: [0,1], a,b: [0,1])"""
    # Convert RGB to BGR for OpenCV
    bgr = rgb[..., ::-1]  # RGB to BGR
    
    # Convert to LAB using TensorFlow
    # Normalize inputs to [0,1]
    lab = tf.image.rgb_to_lab(rgb)
    
    # Normalize to [0,1] ranges
    l = lab[..., 0:1] / 100.0  # L from [0,100] to [0,1]
    a = (lab[..., 1:2] + 128.0) / 255.0  # a from [-128,127] to [0,1]
    b = (lab[..., 2:3] + 128.0) / 255.0  # b from [-128,127] to [0,1]
    
    return np.concatenate([l, a, b], axis=-1)

def numpy_lab_normalized_to_rgb_clipped(lab_norm):
    """Convert normalized LAB values back to RGB and clip to valid range"""
    # Denormalize from [0,1] back to LAB ranges
    l = lab_norm[..., 0:1] * 100.0  # L from [0,1] to [0,100] 
    a = lab_norm[..., 1:2] * 255.0 - 128.0  # a from [0,1] to [-128,127]
    b = lab_norm[..., 2:3] * 255.0 - 128.0  # b from [0,1] to [-128,127]
    
    # Recombine channels
    lab = np.concatenate([l, a, b], axis=-1)
    
    # Convert LAB to RGB using TensorFlow
    rgb = tf.image.lab_to_rgb(lab)
    
    return np.array(rgb)

def _get_model() -> tf.keras.Model:
    global _model
    print(f"[*] Looking for model from {_MODEL_PATH}")
    print(f"[*] Model file exists? {os.path.exists(_MODEL_PATH)}")
    print(f"[*] Current directory: {os.getcwd()}")
    print(f"[*] Model directory contents: {os.listdir(os.path.dirname(_MODEL_PATH))}")
    
    if _model is None:
        # We'll only use the saved model approach - no fallback to training
        try:
            # Make sure the ColorCastRemoval class is properly registered
            print(f"[*] Loading model with ColorCastRemoval class: {ColorCastRemoval}")
            
            # Custom objects dict to help with loading
            custom_objects = {
                'ColorCastRemoval': ColorCastRemoval
            }
            
            # Load the model with custom objects
            _model = tf.keras.models.load_model(
                _MODEL_PATH, 
                custom_objects=custom_objects
            )
            print(f"[*] Successfully loaded Keras model from {_MODEL_PATH}")
        except Exception as e:
            print(f"[!] Error loading model: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model from {_MODEL_PATH}: {str(e)}")
    return _model


def remove_color_cast(
    image_bytes: bytes
) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0

    lab_norm = rgb_to_lab_normalized(arr)

    inp = np.expand_dims(lab_norm, axis=0)       # shape (1,H,W,3)
    model = _get_model()
    
    # Call the model safely
    try:
        outputs = model(inp, training=False)
        if isinstance(outputs, tuple) and len(outputs) >= 1:
            out_lab_norm = outputs[0]
        else:
            out_lab_norm = outputs  # Handle case where model just returns one tensor
        
        out_lab_norm = out_lab_norm.numpy()[0]
        out_rgb = numpy_lab_normalized_to_rgb_clipped(out_lab_norm)
        out_rgb = np.clip(out_rgb, 0.0, 1.0)

        out_img = (out_rgb * 255).astype(np.uint8)
        pil_out = Image.fromarray(out_img)
        buf = io.BytesIO()
        pil_out.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"[!] Error during inference: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed during inference: {str(e)}")


def edit_image(
    image_bytes: bytes,
    brightness: float = 0.0,   # -100 to +100
    contrast: float = 0.0,     # -100 to +100
    saturation: float = 0.0,   # -100 to +100
    temperature: float = 0.0   # -100 to +100
) -> bytes:
    

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0

    # --- Apply brightness
    arr = np.clip(arr * (1 + brightness / 100.0), 0, 1)

    # --- Apply contrast
    arr = np.clip((arr - 0.5) * (1 + contrast / 100.0) + 0.5, 0, 1)

    # --- Apply saturation using OpenCV
    hsv = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + saturation / 100.0), 0, 255)
    arr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # --- Apply temperature: shift red and blue channels
    temp_shift = temperature / 100.0
    arr[..., 0] = np.clip(arr[..., 0] + temp_shift * 0.1, 0, 1)  # Red channel
    arr[..., 2] = np.clip(arr[..., 2] - temp_shift * 0.1, 0, 1)  # Blue channel

    # --- Encode back to bytes
    out_img = (arr * 255).astype(np.uint8)
    pil_out = Image.fromarray(out_img)
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return buf.getvalue()
