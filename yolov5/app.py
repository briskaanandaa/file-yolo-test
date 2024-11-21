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

# Pemetaan indeks ke solusi penyakit
disease_solutions = {
    0: {  
    "name": "Daun Bercak",  
    "solution": "Gunakan fungisida berbasis tembaga dengan konsentrasi 1-2 gram per liter air untuk mengendalikan infeksi jamur. Semprotkan larutan ini secara merata pada daun yang menunjukkan gejala bercak. Selain itu, pastikan pH larutan AB Mix berada pada kisaran 5.5–6.0 untuk membantu penyerapan nutrisi dengan optimal. Hindari rockwool yang terlalu basah dengan mengurangi frekuensi penyiraman dan meningkatkan ventilasi di sekitar area tanaman. Lakukan pembersihan area tanam untuk menghindari penyebaran spora jamur yang dapat menginfeksi tanaman lain."  
},  
1: {  
    "name": "Daun Busuk",  
    "solution": "Pastikan sistem drainase berfungsi dengan baik agar air tidak tergenang di sekitar akar, karena kondisi terlalu lembap dapat memicu pembusukan. Hindari overwatering dengan memantau kebutuhan air tanaman secara berkala dan hanya menyiram ketika rockwool mulai mengering. Tingkatkan sirkulasi udara di sekitar tanaman menggunakan kipas kecil atau ventilasi tambahan untuk mengurangi kelembapan. Stabilkan pH larutan AB Mix pada kisaran 5.8–6.2 agar nutrisi tetap seimbang dan mendukung pemulihan tanaman. Jika ada bagian daun yang sudah rusak parah, potong dan buang untuk mencegah infeksi menyebar lebih luas."  
},  
2: {  
    "name": "Sehat",  
    "solution": "Tidak ada tindakan khusus yang diperlukan, tetapi tetap lakukan pemeliharaan rutin. Jaga pH larutan AB Mix pada kisaran 5.5–6.0 untuk memastikan tanaman tetap menerima nutrisi secara optimal. Periksa kelembapan rockwool secara berkala untuk memastikan media tetap lembap tetapi tidak terlalu basah. Bersihkan area tanam dari kotoran atau sisa daun yang gugur untuk menjaga kebersihan dan mencegah potensi munculnya penyakit. Lakukan inspeksi rutin terhadap daun dan batang tanaman untuk mendeteksi gejala penyakit sejak dini."  
}  
}

# Fungsi untuk menggambar bounding box pada gambar
def draw_bounding_boxes(img, detections):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for obj in detections:
        x_center = int(obj['xcenter'])
        y_center = int(obj['ycenter'])
        width = int(obj['width'])
        height = int(obj['height'])
        index = int(obj['class'])  # Indeks kelas
        confidence = round(obj['confidence'], 3)

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Gambar bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{disease_solutions[index]['name']} {confidence:.3f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
    
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    results = model(img)

    # Mengambil deteksi sebagai format dictionary
    detections = results.pandas().xywh[0].to_dict(orient="records")
    for det in detections:
        class_index = int(det['class'])
        det['name'] = disease_solutions[class_index]["name"]
        det['solution'] = disease_solutions[class_index]["solution"]

    img_with_bboxes = draw_bounding_boxes(img, detections)

    # Menyimpan gambar dengan bounding box ke buffer untuk dikirim sebagai base64
    buffered = BytesIO()
    img_with_bboxes.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Mengembalikan hasil deteksi dan solusi
    return jsonify({
        'detections': detections,
        'image': img_str
    })

if __name__ == "__main__":
    app.run(debug=True)
