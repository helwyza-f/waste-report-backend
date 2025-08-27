from ultralytics import YOLO
import torch

def main():
    # cek apakah GPU tersedia
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training akan dijalankan di device: {device}")

    # load model YOLOv8 (misalnya v8s untuk cepat, bisa ganti v8m/l/x)
    model = YOLO("yolov8s.pt")  

    # training
    model.train(
        data="datasets/data.yaml",   # path ke config dataset YAML kamu
        epochs=50,          # jumlah epoch
        imgsz=640,          # ukuran gambar
        batch=16,           # batch size
        device=device,      # pakai GPU kalau ada
        workers=4           # jumlah workers untuk dataloader
    )

if __name__ == "__main__":
    main()
