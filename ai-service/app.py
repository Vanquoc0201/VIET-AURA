# app.py

import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)
CORS(app) # Bật CORS để cho phép các trang web khác gọi API này

# --- Các đường dẫn và cấu hình ---
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'viet_culture_model.keras')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.txt')
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Tải Model và Tên Lớp (chỉ tải một lần khi server khởi động) ---
print("--- Đang tải model, vui lòng chờ... ---")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("--- Model đã được tải thành công! Server sẵn sàng. ---")
except Exception as e:
    print(f"LỖI: Không thể tải model. Chi tiết: {e}")
    model = None
    class_names = []


# --- Hàm xử lý ảnh đầu vào ---
def preprocess_image(base64_string):
    # 1. Decode chuỗi base64 thành bytes
    img_bytes = base64.b64decode(base64_string)
    # 2. Mở ảnh từ bytes
    img = Image.open(io.BytesIO(img_bytes))
    # 3. Đảm bảo ảnh có 3 kênh màu (RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # 4. Resize ảnh về kích thước model yêu cầu
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    # 5. Chuyển ảnh thành mảng numpy
    img_array = tf.keras.utils.img_to_array(img)
    # 6. Mở rộng chiều để tạo thành một "batch" chứa 1 ảnh
    img_array = np.expand_dims(img_array, axis=0)
    # 7. Chuẩn hóa ảnh theo yêu cầu của MobileNetV2
    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return processed_img


# --- Định nghĩa API Endpoint "/predict" ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded!'}), 500

    try:
        # 1. Lấy dữ liệu JSON từ request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Missing "image" key in request body'}), 400
        
        # 2. Lấy chuỗi base64 của ảnh
        base64_img = data['image']
        
        # 3. Tiền xử lý ảnh
        processed_image = preprocess_image(base64_img)
        
        # 4. Đưa ảnh vào model để dự đoán
        predictions = model.predict(processed_image)
        
        # 5. Xử lý kết quả
        # Lấy chỉ số của lớp có xác suất cao nhất
        predicted_class_index = np.argmax(predictions[0])
        # Lấy tên lớp từ chỉ số đó
        predicted_class_name = class_names[predicted_class_index]
        # Lấy độ tin cậy (xác suất)
        confidence = float(predictions[0][predicted_class_index])
        
        # 6. Trả về kết quả dưới dạng JSON
        return jsonify({
            'object': predicted_class_name,
            'confidence': round(confidence, 4) # Làm tròn cho đẹp
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


# --- Chạy server ---
if __name__ == '__main__':
    # Chạy trên tất cả các địa chỉ IP của máy, cổng 5000
    app.run(host='0.0.0.0', port=5000, debug=True)