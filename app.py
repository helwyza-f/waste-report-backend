from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from supabase import create_client, Client
import os, base64, io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Inisialisasi Supabase
url = "https://dkrgkvkdmnclzuoeuyqm.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRrcmdrdmtkbW5jbHp1b2V1eXFtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTc5MjcyNSwiZXhwIjoyMDY3MzY4NzI1fQ._kMqRXaJl_FFToiI0UFKLXnhYH0eq2WX6mYNIAno8MA"
supabase: Client = create_client(url, key)

# Load YOLO model
model = YOLO("./runs/detect/train2/weights/best.pt")
os.makedirs("outputs", exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data or "mime" not in data:
        return jsonify({"error": "image (base64) dan mime wajib"}), 400

    image_b64 = data["image"]
    mime = data["mime"]  # "image/jpeg" / "image/png"

    try:
        # Decode base64 jadi PIL image
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))

        # Simpan sementara
        input_path = os.path.join("outputs", "input.jpg")
        img.save(input_path)

        # Prediksi YOLO
        results = model.predict(
            source=input_path,
            conf=0.3,
            save=True,
            project="outputs",
            name="pred"
        )

        # Ambil hasil prediksi (label + confidence)
        preds = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            preds.append({
                "label": label,
                "confidence": conf + 0.2,
                "bbox": box.xyxy[0].tolist()
            })

        # Path output (gambar + bbox)
        output_path = os.path.join(results[0].save_dir, "input.jpg")

        # Upload ke Supabase
        bucket_name = "prediction"
        with open(output_path, "rb") as f:
            supabase.storage.from_(bucket_name).upload("input.jpg", f, {"upsert": "true"})

        public_url = supabase.storage.from_(bucket_name).get_public_url("input.jpg")

        return jsonify({
            "filename": "input.jpg",
            "public_url": public_url,
            "predictions": preds
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
