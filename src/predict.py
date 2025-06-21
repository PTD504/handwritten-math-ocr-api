import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from config import config

# Change this to test with other type of model (model_swin or model_res18trans)
from model import FormulaRecognitionModel

# === Setting ===
model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
image_test_path = "Path/to/your/test/image.png"
vocab_path = os.path.join(config.checkpoint_dir, 'vocab.json')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load vocab ===
with open(vocab_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    vocab = data['vocab']
    idx2token = {v: k for k, v in vocab.items()}

# === Load model ===
# # In case you used model from checkpoint
model = FormulaRecognitionModel(len(vocab)).to(device)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# # In case you used MLflow to log model
# model = torch.load(model_path, map_location=device, weights_only=False)
# model.eval()

# === Image preprocessing ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config.img_h, config.img_w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)

# === Predict ===
def predict(image_tensor, max_len=150):
    sos_token = vocab[config.sos_token]
    eos_token = vocab[config.eos_token]

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)
        output_seq = [sos_token]

        for _ in range(max_len):
            input_seq = torch.tensor(output_seq, dtype=torch.long, device=device).unsqueeze(0)
            logits = model.decoder(encoder_out, input_seq)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            output_seq.append(next_token)

            if next_token == eos_token:
                break

    return [idx2token[idx] for idx in output_seq[1:-1]]

# === Test ===
if __name__ == "__main__":
    image_tensor = preprocess_image(image_test_path)
    tokens = predict(image_tensor)
    latex_output = ' '.join(tokens)

    print("Predicted LaTeX:", latex_output)