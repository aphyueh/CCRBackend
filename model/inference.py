# model/inference.py
import cv2
import io
import numpy as np
import os
from PIL import Image, ImageEnhance
import tensorflow as tf

from .model import ColorCastRemoval
from .utils import  rgb_to_lab_normalized, numpy_lab_normalized_to_rgb_clipped

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")
_model = None

def initialize_model():
    global model
    try:
        model = _get_model()
        print("[INFO] Model loaded successfully")
    except Exception as e:
        print(f"[!] Model failed to load at startup: {e}")

def _get_model() -> tf.keras.Model:
    global _model
    print(f"[*] Looking for model from {_MODEL_PATH}")
    print(f"[*] Model file exists? {os.path.exists(_MODEL_PATH)}")
    print(f"[*] Current directory: {os.getcwd()}")
    print(f"[*] Model directory contents: {os.listdir(os.path.dirname(_MODEL_PATH))}")
    
    if _model is None:
        if os.path.exists(_MODEL_PATH):
            # 1) Load the full .keras model
            _model = tf.keras.models.load_model(
                _MODEL_PATH,
                custom_objects={"ColorCastRemoval": ColorCastRemoval}
            )
            print(f"[*] Loaded Keras model from {_MODEL_PATH}")
        else:
            # 2) Fallback: build from scratch + checkpoint
            _model = ColorCastRemoval()
            ckpt = tf.train.Checkpoint(model=_model)
            latest = tf.train.latest_checkpoint("./ml/checkpoints")
            if latest:
                ckpt.restore(latest).expect_partial()
                print(f"[*] Restored from checkpoint: {latest}")
            else:
                print("[!] No checkpoint found; using untrained model")
    return _model


def remove_color_cast(
    image_bytes: bytes
) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0

    lab_norm = rgb_to_lab_normalized(arr)
    inp = np.expand_dims(lab_norm, axis=0)       # shape (1,H,W,3)

    if model is None:
        raise RuntimeError("Model not loaded")
    
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