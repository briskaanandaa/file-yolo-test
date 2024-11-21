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
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Asus/Downloads/file-yolo-test/yolov5/runs/train/exp4/weights/best.pt')

# Pemetaan indeks ke solusi penyakit
disease_solutions = {
    0: {"name": "Daun Bercak", "solution": "Daun selada yang mengalami bercak sering kali disebabkan oleh ketidakseimbangan pH atau kekurangan nutrisi tertentu. Ketika pH media tanam berada di bawah angka 5.5, dapat terjadi peningkatan kelarutan logam berat seperti mangan yang berbahaya bagi tanaman dan menyebabkan bercak pada daun. Selain itu, kekurangan nutrisi seperti kalium (K) dan magnesium (Mg) dapat memperburuk kondisi tersebut. Kalium berperan penting dalam menjaga keseimbangan air dalam sel tanaman dan meningkatkan ketahanan terhadap stres lingkungan, sedangkan magnesium merupakan komponen penting dalam fotosintesis. Oleh karena itu, untuk memperbaiki daun yang bercak, langkah pertama adalah memperbaiki pH media tanam agar berada dalam kisaran 5.8 hingga 6.5, yang merupakan pH optimal bagi selada. Penggunaan AB Mix yang mengandung kalium dalam jumlah yang cukup sangat disarankan untuk membantu tanaman mengatasi kekurangan kalium. Selain itu, penambahan magnesium dalam nutrisi dapat mendukung pertumbuhan daun yang sehat dan memperbaiki bercak yang muncul akibat kekurangan kedua unsur tersebut."},
    1: {"name": "Daun Busuk", "solution": "Daun selada yang mengalami pembusukan dapat disebabkan oleh beberapa faktor, termasuk ketidakseimbangan pH dan kekurangan nutrisi tertentu. pH yang terlalu tinggi (di atas 7.5) dapat mengurangi kemampuan tanaman dalam menyerap nutrisi penting, seperti fosfor dan kalsium. Kalsium, khususnya, sangat penting dalam pembentukan dinding sel dan penguatan jaringan tanaman, sehingga kekurangan kalsium dapat menyebabkan jaringan daun menjadi lebih rentan terhadap pembusukan. Untuk memperbaiki daun yang busuk, penting untuk menyesuaikan pH media tanam agar berada dalam kisaran 5.8 hingga 6.5. Penggunaan AB Mix yang mengandung kalsium akan membantu memenuhi kebutuhan kalsium tanaman, mencegah pembusukan akibat kekurangan unsur tersebut. Selain itu, dengan memperbaiki pH dan memastikan kalsium tercukupi, tanaman akan memiliki ketahanan yang lebih baik terhadap penyakit dan kerusakan jaringan, yang akan membantu memperbaiki kondisi daun yang busuk."},
    2: {"name": "Sehat", "solution": "Daun selada yang sehat menunjukkan bahwa kondisi pH dan ketersediaan nutrisi berada dalam keseimbangan yang optimal. pH media tanam yang ideal untuk selada berada dalam kisaran 5.8 hingga 6.5. Pada kisaran pH ini, tanaman dapat menyerap unsur hara dengan efisien, mendukung pertumbuhan daun yang optimal. Untuk menjaga daun tetap sehat, penting untuk memastikan bahwa AB Mix yang digunakan mengandung keseimbangan antara unsur hara utama seperti nitrogen (N), fosfor (P), dan kalium (K), serta mikro nutrisi seperti magnesium (Mg) dan kalsium (Ca). Nitrogen mendukung pertumbuhan daun yang cepat dan hijau, fosfor berperan dalam perkembangan akar dan sistem perakaran yang sehat, sedangkan kalium membantu dalam pengaturan air dan ketahanan terhadap stres. Kalsium mendukung integritas sel tanaman dan memperkuat jaringan daun, sedangkan magnesium penting untuk proses fotosintesis yang optimal. Penggunaan AB Mix dengan komposisi nutrisi yang tepat akan mendukung pertumbuhan selada yang sehat dan optimal, menjaga tanaman tetap kuat dan produktif."}
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

# Route utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Ambil nilai pH dan nutrisi (TDS) dari form
    ph = float(request.form.get('ph', 0))
    nutrient = int(request.form.get('nutrient', 0))

    # Baca gambar dan lakukan prediksi
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    results = model(img)

    # Proses hasil deteksi
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    for det in detections:
        class_index = int(det['class'])
        det['name'] = disease_solutions[class_index]["name"]

        # Penyesuaian solusi berdasarkan pH
        det['solution'] = disease_solutions[class_index]["solution"]
        if ph < 5.5:
            det['solution'] += " pH air pada tanaman terlalu rendah (asam), tanaman selada akan kesulitan menyerap beberapa unsur hara penting, seperti kalsium, magnesium, dan fosfor. Gejala yang sering muncul adalah daun yang berwarna kuning, serta pertumbuhan akar yang terhambat. pH yang terlalu asam juga dapat meningkatkan kelarutan unsur hara yang berbahaya bagi tanaman, seperti logam berat, yang bisa menyebabkan keracunan. Pastikan bahwa keadaan rockwool tanaman telah dialiri oleh sistem dengan baik dan benar."
        elif 5.5 <= ph <= 6.5:
            det['solution'] += " pH air pada tanaman tepat, yaitu di kisaran pH 5,5 hingga 6,5, adalah kondisi ideal untuk pertumbuhan selada. Pada pH ini, tanaman selada dapat menyerap nutrisi dengan efisien, memastikan pertumbuhannya optimal. pH yang tepat juga mendukung perkembangan mikroorganisme tanah yang bermanfaat, yang membantu proses dekomposisi dan ketersediaan nutrisi. Rockwool tanaman telah dialiri oleh sistem dengan baik dan benar."
        else:  # ph > 6.5
            det['solution'] += " pH air pada tanaman terlalu tinggi (alkalis), tanaman selada dapat kesulitan menyerap unsur hara tertentu, seperti besi, mangan, dan boron, yang sangat penting untuk pertumbuhannya. Gejala yang dapat terlihat pada tanaman selada dengan pH air berlebih adalah daun yang menguning, pertumbuhan yang terhambat, dan daunnya tampak kering atau terbakar. Selain itu, pH tinggi dapat menyebabkan penurunan ketersediaan unsur hara lainnya, yang mengarah pada defisiensi nutrisi pada tanaman. Pastikan bahwa keadaan rockwool tanaman telah dialiri oleh sistem dengan baik dan benar."

        # Penyesuaian solusi berdasarkan nutrisi (TDS)
        if nutrient < 800:
            det['solution'] += " Dari data yang didapat tanaman selada kekurangan nutrisi AB Mix, pertumbuhannya bisa terhambat. Nutrisi yang tidak mencukupi dapat menyebabkan gejala seperti daun yang menguning, pertumbuhan yang lambat, dan tanaman menjadi kurus atau lemah. Kekurangan unsur hara tertentu seperti nitrogen (N), fosfor (P), atau kalium (K) dapat mengganggu fungsi metabolisme tanaman, sehingga menyebabkan stres pada tanaman selada. Pastikan bahwa keadaan rockwool tanaman telah dialiri oleh sistem dengan baik dan benar."
        elif 800 <= nutrient <= 1233000:
            det['solution'] += " Berdasarkan Data Yang dimasukkan penggunaan Nutrisi AB Mix yang tepat membantu mendukung pertumbuhan tanaman selada secara optimal. Jika komposisi nutrisi sudah sesuai, tanaman akan tumbuh sehat, dengan daun yang hijau dan segar. Nutrisi yang tepat memastikan bahwa tanaman selada mendapatkan semua elemen penting yang diperlukan untuk fotosintesis dan metabolisme, seperti nitrogen, fosfor, kalium, magnesium, dan kalsium, sehingga meminimalisir potensi penyakit dan defisiensi. Rockwool tanaman telah dialiri oleh sistem dengan baik dan benar."
        else:  # nutrient > 600
            det['solution'] += " Pemberian nutrisi AB Mix yang berlebihan dapat menyebabkan akumulasi garam dalam medium tanam, yang menghambat penyerapan air oleh akar dan bisa menyebabkan keracunan pada tanaman. Gejala yang muncul biasanya adalah daun yang terbakar atau menguning, dan akar yang rusak atau mati. Kelebihan nutrisi juga dapat menurunkan pH media tanam, yang selanjutnya dapat mengganggu keseimbangan unsur hara dan menyebabkan ketidakseimbangan pada tanaman. Pastikan bahwa keadaan rockwool tanaman telah dialiri oleh sistem dengan baik dan benar."

    # Gambar bounding box pada hasil deteksi
    img_with_bboxes = draw_bounding_boxes(img, detections)

    # Konversi gambar ke format base64 untuk ditampilkan
    buffered = BytesIO()
    img_with_bboxes.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return hasil prediksi dan gambar dengan bounding box
    return jsonify({'detections': detections, 'image': img_str})

if __name__ == "__main__":
    app.run(debug=True)
