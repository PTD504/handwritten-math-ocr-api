# Sử dụng base image có hỗ trợ CUDA.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Cài đặt Python 3 (mặc định trên Ubuntu 22.04 là 3.10) và các công cụ cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Xóa các gói không cần thiết để giảm kích thước image
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Sao chép requirements.txt
COPY requirements.txt .

# Thay đổi cách cài đặt PyTorch
# Thêm --index-url để chỉ định kho PyTorch CUDA
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Sao chép mã nguồn và mô hình
COPY ./src ./src
COPY ./trained-model ./trained-model

EXPOSE 8080

# Chạy ứng dụng
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
