from flask import Flask, render_template, request, jsonify
import torch
import io
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import requests

app = Flask(__name__)

# Load model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Asus/Downloads/file-yolo-test/yolov5/runs/train/exp4/weights/best.pt')

# Pemetaan indeks ke solusi penyakit
disease_solutions = {
    0: {"name": "Daun Bercak", "solution": "Daun selada yang mengalami bercak sering kali disebabkan oleh ketidakseimbangan pH atau kekurangan nutrisi tertentu. ..."},
    1: {"name": "Daun Busuk", "solution": "Daun selada yang mengalami pembusukan dapat disebabkan oleh beberapa faktor, termasuk ketidakseimbangan pH dan kekurangan nutrisi tertentu. ..."},
    2: {"name": "Sehat", "solution": "Daun selada yang sehat menunjukkan bahwa kondisi pH dan ketersediaan nutrisi berada dalam keseimbangan yang optimal. ..."}
}

# Fungsi untuk menggambar bounding box
def draw_bounding_boxes(img, detections):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for obj in detections:
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        label = f"{obj['name']} ({obj['confidence']:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Terima file gambar dari form HTML
        file = request.files['image']
        if not file:
            return jsonify({"error": "Tidak ada file yang diunggah"}), 400
        
        # Konversi file menjadi gambar PIL
        img = Image.open(file.stream).convert('RGB')
        
        # Deteksi menggunakan YOLOv5
        results = model(img)
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        
        # Sesuaikan nama dan solusi
        for det in detections:
            class_index = int(det['class'])
            det['name'] = disease_solutions[class_index]["name"]
            det['solution'] = disease_solutions[class_index]["solution"]
        
        # Gambar bounding box
        img_with_bboxes = draw_bounding_boxes(img, detections)
        
        # Konversi ke base64 untuk ditampilkan
        buffered = BytesIO()
        img_with_bboxes.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({'detections': detections, 'image': img_str})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/esp-cam', methods=['GET'])
def esp_cam():
    try:
        # URL streaming ESP CAM
        url = "http://192.168.18.223/capture"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image_bytes = response.content
            img = Image.open(io.BytesIO(image_bytes))
            
            # Deteksi menggunakan YOLOv5
            results = model(img)
            detections = results.pandas().xyxy[0].to_dict(orient="records")
            
            # Sesuaikan nama dan solusi
            for det in detections:
                class_index = int(det['class'])
                det['name'] = disease_solutions[class_index]["name"]
                det['solution'] = disease_solutions[class_index]["solution"]
            
            # Gambar bounding box
            img_with_bboxes = draw_bounding_boxes(img, detections)
            
            # Konversi ke base64
            buffered = BytesIO()
            img_with_bboxes.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return jsonify({'detections': detections, 'image': img_str})
        else:
            return jsonify({"error": "Tidak dapat mengakses ESP CAM"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
