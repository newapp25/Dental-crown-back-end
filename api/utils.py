import cv2
import numpy as np
from fastapi import HTTPException

# Validate uploaded image files
def validate_image(file_bytes):
    try:
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image format")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

# Preprocess images for depth estimation
def preprocess_image(image, target_size=(384, 384)):
    resized_image = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0
    return normalized_image

# Error handler
def handle_error(message, code=400):
    raise HTTPException(status_code=code, detail=message)
