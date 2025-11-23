import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from huggingface_hub import hf_hub_download


# -------------------------------------------------------
# 1. LOAD MODEL FROM HUGGINGFACE
# -------------------------------------------------------
def load_classifier():
    """Download and load the PyTorch model from HuggingFace Hub."""

    model_path = hf_hub_download(
        repo_id="roshanthm/space-vision-model",
        filename="space_classifier.pt"
    )

    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model


# -------------------------------------------------------
# 2. PREPROCESSING
# -------------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])


def preprocess(image):
    """Convert raw image (OpenCV or PIL) into a model-ready tensor."""
    
    if isinstance(image, np.ndarray):  # OpenCV → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension
    return image


# -------------------------------------------------------
# 3. PREDICT SPACE OBJECTS
# -------------------------------------------------------
def predict_space_object(model, image):
    """
    Returns:
        class_name (str)
        confidence (float)
    """

    input_tensor = preprocess(image)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(prob, dim=1)

    # These labels should match your training classes
    class_labels = [
        "Galaxy",
        "Nebula",
        "Star Cluster",
        "Exoplanet Transit",
        "Asteroid",
        "Comet",
        "Supernova Candidate",
        "Unknown Object"
    ]

    class_name = class_labels[predicted.item()]
    confidence = confidence.item()

    return class_name, confidence


# -------------------------------------------------------
# 4. DRAW RESULTS ON IMAGE
# -------------------------------------------------------
def draw_result(image, label, conf):
    """
    Draw predicted label on image frame for display.
    """
    text = f"{label} ({conf*100:.1f}%)"
    
    cv2.putText(image, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image


# -------------------------------------------------------
# 5. MAIN PIPELINE (used by app.py)
# -------------------------------------------------------
def detect(image):
    """
    This function is imported and used in app.py.
    """
    model = load_classifier()
    label, conf = predict_space_object(model, image)
    annotated = draw_result(image.copy(), label, conf)

    return label, conf, annotated
