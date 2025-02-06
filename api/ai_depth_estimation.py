import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

# Load the AI model
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model.eval()

def predict_depth(image):
    if len(image.shape) == 2:  # Grayscale to RGB
        image = np.stack((image,) * 3, axis=-1)

    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
        depth = output.predicted_depth.squeeze().cpu().numpy()

    # Normalize depth map
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth_normalized.tolist()
