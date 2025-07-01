# train.py

import tensorflow as tf
import os

# --- Các tham số cấu hình ---
DATASET_DIR = 'dataset'
MODEL_SAVE_DIR = 'model'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'viet_culture_model.keras')
CLASS_NAMES_PATH = os.path.join(MODEL_SAVE_DIR, 'class_names.txt')

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 15 # Số lần lặp lại quá trình huấn luyện trên toàn bộ dataset

# --- Bước 1: Tải và chuẩn bị dữ liệu ---
print("--- Bắt đầu tải dữ liệu ---")

# Tải dữ liệu huấn luyện từ thư mục, tự động chia 80% cho training
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2, # 20% dữ liệu sẽ được dùng để kiểm thử (validation)
    subset="training",
    seed=123, # Đặt seed để đảm bảo kết quả có thể tái tạo
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical' # Quan trọng: Vì ta có nhiều lớp
)

# Tải 20% dữ liệu còn lại cho việc kiểm thử
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Lấy danh sách tên các lớp (ví dụ: ['ao_dai', 'banh_mi',...])
class_names = train_ds.class_names
print(f"Đã tìm thấy các lớp: {class_names}")

# Tối ưu hóa hiệu suất đọc dữ liệu
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("--- Tải dữ liệu hoàn tất ---")


# --- Bước 2: Xây dựng Model bằng Transfer Learning ---
print("--- Bắt đầu xây dựng model ---")

# Tải mô hình MobileNetV2 đã được huấn luyện trước trên bộ dữ liệu ImageNet
# include_top=False: Bỏ đi lớp phân loại cuối cùng của model gốc
# để chúng ta có thể thêm lớp phân loại của riêng mình.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# Đóng băng các lớp của model gốc.
# Chúng ta không muốn huấn luyện lại chúng, chỉ muốn tận dụng các đặc trưng đã học.
base_model.trainable = False

# Xây dựng model mới bằng cách thêm các lớp của chúng ta vào trên model gốc
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
# Chuẩn hóa đầu vào theo yêu cầu của MobileNetV2
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
# Lớp model gốc
x = base_model(x, training=False)
# Thêm các lớp xử lý của chúng ta
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x) # Giúp tránh overfitting
# Lớp đầu ra cuối cùng với số nơ-ron bằng số lớp của chúng ta
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

print("--- Xây dựng model hoàn tất ---")


# --- Bước 3: Biên dịch và Huấn luyện Model ---
print("--- Bắt đầu biên dịch và huấn luyện ---")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Sử dụng loss này cho bài toán phân loại đa lớp
    metrics=['accuracy']
)

# In cấu trúc model để kiểm tra
model.summary()

# Bắt đầu huấn luyện
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("--- Huấn luyện hoàn tất ---")


# --- Bước 4: Lưu Model và Tên Lớp ---
print("--- Bắt đầu lưu model và tên các lớp ---")

# Tạo thư mục 'model' nếu nó chưa tồn tại
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Lưu model đã huấn luyện
model.save(MODEL_SAVE_PATH)
print(f"Model đã được lưu tại: {MODEL_SAVE_PATH}")

# Lưu lại tên các lớp để ứng dụng API có thể sử dụng
with open(CLASS_NAMES_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(class_names))
print(f"Tên các lớp đã được lưu tại: {CLASS_NAMES_PATH}")

print("--- HOÀN THÀNH GIAI ĐOẠN 3! ---")