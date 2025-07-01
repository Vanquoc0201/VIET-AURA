# predict.py
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

# Tải model MobileNetV2 pretrained từ ImageNet
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Tiền xử lý input
def preprocess_image(base64_str):
    image_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(image_data)).convert('RGB').resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Dự đoán
def predict_image(base64_img):
    try:
        img_tensor = preprocess_image(base64_img)
        preds = model.predict(img_tensor)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
        return {
            "object": decoded[1],      # label
            "confidence": float(decoded[2])
        }
    except Exception as e:
        return {"error": str(e)}
