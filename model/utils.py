# model/utils.py
import numpy as np
from skimage import color
import tensorflow as tf
import warnings # Import warnings module

# --- Numpy function for RGB to Normalized LAB ---
# This function seems okay, input RGB is clipped before conversion.
def rgb_to_lab_normalized(rgb_image_np):
    """Convert RGB [0,1] numpy array to normalized LAB [0,1] numpy array."""
    # Input should be numpy array (single image H,W,C or batch N,H,W,C)
    was_single_image = False
    if rgb_image_np.ndim == 3:
        rgb_image_np = np.expand_dims(rgb_image_np, axis=0)
        was_single_image = True

    batch_lab = []
    for img_rgb in rgb_image_np:
        # Ensure input is float64 for skimage and clipped
        img_rgb_clipped = np.clip(img_rgb, 0.0, 1.0).astype(np.float64)
        # Perform conversion
        lab = color.rgb2lab(img_rgb_clipped) # Input range [0, 1] -> Output L[0,100], a/b approx [-128, 127]

        # Normalize L from [0, 100] to [0, 1]
        lab[..., 0] = lab[..., 0] / 100.0
        # Normalize a and b from approx [-128, 127] to [0, 1] (center 0 maps to 0.5)
        lab[..., 1:] = (lab[..., 1:] + 128.0) / 255.0

        # Clip normalized LAB to ensure valid range [0, 1] after normalization math
        lab = np.clip(lab, 0.0, 1.0)
        batch_lab.append(lab)

    lab_batch = np.stack(batch_lab, axis=0)

    if was_single_image:
        lab_batch = lab_batch[0] # Remove batch dim if input was single image

    return lab_batch.astype(np.float32) # Return as float32


# --- Numpy function for Normalized LAB to RGB (with added clipping) ---
# This is the function that will be wrapped by tf.py_function
def numpy_lab_normalized_to_rgb_clipped(lab_normalized_np):
    """
    Convert normalized LAB [0,1] numpy array back to RGB [0,1] numpy array.
    Includes clipping steps to potentially reduce skimage warnings.
    """
    # --- Clipping Step 1: Clip input normalized LAB ---
    # Ensure the input is strictly within the [0, 1] range expected
    lab_normalized_np = np.clip(lab_normalized_np, 0.0, 1.0)
    # --- End Clipping Step 1 ---

    was_single_image = False
    # Use the (potentially clipped) numpy array for subsequent operations
    if lab_normalized_np.ndim == 3:
        lab_normalized_np = np.expand_dims(lab_normalized_np, axis=0)
        was_single_image = True

    batch_rgb = []
    # Iterate using the numpy array
    for lab_norm in lab_normalized_np:
        # Now lab_norm is guaranteed to be a numpy array within [0, 1]
        lab = lab_norm.astype(np.float64)
        denorm_lab = np.copy(lab)

        # Denormalize L from [0, 1] to [0, 100]
        denorm_lab[..., 0] = denorm_lab[..., 0] * 100.0
        # Denormalize a and b from [0, 1] to approx [-128, 127]
        denorm_lab[..., 1:] = denorm_lab[..., 1:] * 255.0 - 128.0

        # --- Clipping Step 2: Clip denormalized L channel ---
        # Clip L to its valid range [0, 100] before lab2rgb conversion
        denorm_lab[..., 0] = np.clip(denorm_lab[..., 0], 0.0, 100.0)
       
        rgb = color.lab2rgb(denorm_lab) # Output range [0, 1]

        # --- Clipping Step 3: Clip final RGB output (Important!) ---
        # Ensure the final RGB values are strictly within [0, 1]
        rgb = np.clip(rgb, 0.0, 1.0)
        # --- End Clipping Step 3 ---

        batch_rgb.append(rgb)

    rgb_batch = np.stack(batch_rgb, axis=0)

    if was_single_image:
        rgb_batch = rgb_batch[0] # Remove batch dim if input was single image

    return rgb_batch.astype(np.float32) # Return as float32


# --- TensorFlow Wrappers (using tf.py_function) ---

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)])
def tf_rgb_to_lab_normalized(rgb_image_tensor):
    """TensorFlow wrapper for RGB to normalized LAB conversion."""
    # This wrapper uses the original rgb_to_lab_normalized numpy function
    lab_tensor = tf.py_function(
        func=rgb_to_lab_normalized,
        inp=[rgb_image_tensor],
        Tout=tf.float32
    )
    # Set shape information crucial for graph execution
    # Use None for dynamic dimensions (batch, height, width)
    lab_tensor.set_shape([None, None, None, 3])
    return lab_tensor

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)])
def tf_lab_normalized_to_rgb(lab_norm_tensor):
    """
    TensorFlow wrapper for normalized LAB to RGB conversion.
    Uses the numpy function with added clipping.
    """
    # This wrapper now calls the numpy function with added clipping
    rgb_tensor = tf.py_function(
        func=numpy_lab_normalized_to_rgb_clipped, # Call the modified numpy function
        inp=[lab_norm_tensor],
        Tout=tf.float32
    )
    # Set shape information crucial for graph execution
    rgb_tensor.set_shape([None, None, None, 3]) # Use None for dynamic dimensions
    return rgb_tensor
