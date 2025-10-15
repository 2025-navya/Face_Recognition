import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import pickle
import os
import matplotlib.pyplot as plt

# Load model and label map
model = load_model("../models/face_cnn_model.h5")
with open("../models/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

def predict_face(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]

    predicted_index = np.argmax(preds)
    predicted_label = label_map[predicted_index]
    confidence = preds[predicted_index]

    # Prepare prediction output
    result = {
        "label": predicted_label,
        "confidence": float(confidence),
        "all_predictions": {
            label_map[i]: float(preds[i]) for i in range(len(preds))
        }
    }

    return result
