from flask import Flask, render_template, request, jsonify
import torch
import io
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# Load model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:/Users/Asus/Downloads/file-yolo-test/yolov5/runs/train/exp4/weights/best.pt')

# Fungsi untuk menggambar bounding box pada gambar
def draw_bounding_boxes(img, detections):
    # Konversi gambar PIL ke format OpenCV (BGR)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for obj in detections:
        x_center = int(obj['xcenter'])
        y_center = int(obj['ycenter'])
        width = int(obj['width'])
        height = int(obj['height'])
        name = obj['name']
        confidence = round(obj['confidence'], 3)

        # Menghitung koordinat untuk bounding box
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Menggambar bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} {confidence:.3f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mengkonversi kembali gambar OpenCV ke PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img_pil

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menangani upload gambar atau deteksi melalui kamera
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Membaca gambar dari file yang diupload
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    
    # Menggunakan model YOLO untuk mendeteksi objek
    results = model(img)
    
    # Mengambil deteksi sebagai format dictionary
    detections = results.pandas().xywh[0].to_dict(orient="records")

    # Gambar bounding box pada gambar yang telah diproses
    img_with_bboxes = draw_bounding_boxes(img, detections)

    # Menyimpan gambar dengan bounding box ke buffer untuk dikirim sebagai base64
    buffered = BytesIO()
    img_with_bboxes.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Mengembalikan hasil deteksi dan gambar yang telah diproses
    return jsonify({
        'detections': detections,
        'image': img_str
    })

if __name__ == "__main__":
    app.run(debug=True)
