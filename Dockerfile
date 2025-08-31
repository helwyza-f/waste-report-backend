# Gunakan base image PyTorch resmi yang sudah termasuk CUDA dan cuDNN
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install dependensi sistem yang diperlukan oleh OpenCV
# pytorch base image tidak mencakup ini
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy sisa file aplikasi
COPY . .

# Download model YOLOv8 (Ini penting, karena model harus ada di dalam image Docker)
RUN wget -O /app/best.pt https://huggingface.co/username/model-repo/resolve/main/best.pt

# Expose port untuk Flask
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "app.py"]