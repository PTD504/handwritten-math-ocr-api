import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from config import config

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Tiền xử lý ảnh đầu vào để phù hợp với mô hình nhận dạng công thức toán học.
    
    Args:
        image (PIL.Image): Ảnh đầu vào (có thể là RGB hoặc grayscale).
    
    Returns:
        torch.Tensor: Tensor ảnh đã được tiền xử lý (shape: [1, 1, img_h, img_w]).
    """
    # Áp dụng transform: ToTensor và Normalize
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config.img_h, config.img_w)),
        transforms.ToTensor(),  # Chuyển thành tensor [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Chuẩn hóa
    ])
    
    img_tensor = transform(image)  # Shape: [1, img_h, img_w]
    img_tensor = img_tensor.unsqueeze(0)  # Thêm batch dimension: [1, 1, img_h, img_w]
    
    return img_tensor