import cv2
import numpy as np
from pathlib import Path


def load_image(image_path):
    """Loads an image in grayscale."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies CLAHE to enhance contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def resize_image(img, size=(224, 224)):
    """Resizes image to match CNN input dimensions."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def preprocess_image(image_path, output_path):
    """Loads, enhances, resizes, and saves the preprocessed image."""
    try:
        img = load_image(image_path)
        img = apply_clahe(img)
        img = resize_image(img)
        cv2.imwrite(str(output_path), img)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False