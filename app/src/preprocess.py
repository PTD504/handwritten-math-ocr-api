import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from config import config

def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config.img_h, config.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor