from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
import torch.nn.functional as F

# My hugging face cat emotion classifier model
# It is automatically installed from hugging face model hub when first used.
MODEL_PATH = r"semihdervis/cat-emotion-classifier" 

def load_model():
    """
    Loads the ViT model and feature extractor from the specified path.
    """
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH)
    return model, feature_extractor

def preprocess_image(image, feature_extractor):
    """
    Converts the image to RGB format and extracts features using the feature extractor.
    """
    image = image.convert("RGB")  # Ensure image is in RGB format
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def predict(image):
    """
    Predicts the emotion from the uploaded image using the ViT model.
    Returns a dictionary of label and confidence scores.
    """
    # Load model and feature extractor
    model, feature_extractor = load_model()

    # Preprocess image
    inputs = preprocess_image(image, feature_extractor)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to the same device as the model

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get confidence scores for all labels
    confidence_scores = F.softmax(logits, dim=-1)[0] * 100  # Convert to percentage
    label_confidence = {model.config.id2label[i]: score.item() for i, score in enumerate(confidence_scores)}
    return label_confidence