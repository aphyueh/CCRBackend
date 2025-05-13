# model/inference.py
import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf

from .model import ColorCastRemoval
from .utils import rgb_to_lab_normalized, numpy_lab_normalized_to_rgb_clipped

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")
_model = None

def _get_model() -> tf.keras.Model:
    global _model
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

def remove_colour_cast(
    image_bytes: bytes,
    brightness_pct: float = 100.0,
    noise_pct:      float = 0.0,
    contrast_pct:   float = 100.0
) -> bytes:
    # --- 1) Decode to float‐RGB [0,1]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0

    # --- 2) Apply user‐supplied brightness
    arr = arr * (brightness_pct / 100.0)

    # --- 3) Apply contrast adjustment
    #     scale around midpoint 0.5
    arr = (arr - 0.5) * (contrast_pct / 100.0) + 0.5

    # --- 4) Add noise if requested
    if noise_pct > 0.0:
        sigma = noise_pct / 100.0
        noise = np.random.normal(scale=sigma, size=arr.shape).astype(np.float32)
        arr = arr + noise

    # --- 5) Clip back into [0,1]
    arr = np.clip(arr, 0.0, 1.0)

    # --- 6) RGB → normalized LAB
    lab_norm = rgb_to_lab_normalized(arr)

    # --- 7) Model inference
    inp = np.expand_dims(lab_norm, axis=0)       # shape (1,H,W,3)
    model = _get_model()
    out_lab_norm, _ = model(inp, training=False)[:2]

    # --- 8) LAB → RGB via your utils helper
    out_lab_norm = out_lab_norm.numpy()[0]
    out_rgb = numpy_lab_normalized_to_rgb_clipped(out_lab_norm)
    out_rgb = np.clip(out_rgb, 0.0, 1.0)

    # --- 9) Encode back to PNG bytes
    out_img = (out_rgb * 255).astype(np.uint8)
    pil_out = Image.fromarray(out_img)
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return buf.getvalue()