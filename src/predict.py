import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from model_res18trans import FormulaRecognitionModel
from config import config

# === Cấu hình ===
model_path = os.path.join(os.path.dirname(os.getcwd()), 'checkpoints', 'best_model.pth')
image_test_path = "Path/to/your/test/image.png"  # Thay đổi đường dẫn tới ảnh test
vocab_path = os.path.join(os.path.dirname(os.getcwd()), 'checkpoints', 'vocab.json')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# === Load vocab ===
with open(vocab_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    vocab = data['vocab']
    idx2token = {v: k for k, v in vocab.items()}

# === Load model ===
model = FormulaRecognitionModel(len(vocab)).to(device)
# checkpoint = torch.load(model_path, map_location=device)
# model = torch.load(checkpoint['model_state_dict])
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# === Xử lý ảnh ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config.img_h, config.img_w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0) # [1, 1, H, W]

# === Dự đoán ===
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

    return [idx2token[idx] for idx in output_seq[1:-1]]  # Bỏ <sos> và <eos>

def predict_with_topk(model, image_tensor, vocab, idx2token, device, k=5, max_len=150):
    model.eval()
    image_tensor = image_tensor.to(device)  # [1, 1, H, W]
    
    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)  # [1, seq_len, d_model]
        tgt = torch.tensor([[vocab[config.sos_token]]], device=device)  # [1, 1]

        for i in range(max_len):
            output = model.decoder(encoder_out, tgt)  # [1, seq_len, vocab_size]
            logits = output[:, -1, :]  # Lấy logits ở bước hiện tại [1, vocab_size]
            probs = F.softmax(logits, dim=-1)

            # Lấy top-k token
            topk_probs, topk_indices = probs.topk(k, dim=-1)
            topk_probs = topk_probs.squeeze().cpu().numpy()
            topk_indices = topk_indices.squeeze().cpu().numpy()

            # In ra top-k token với xác suất
            print(f"Step {i+1}:")
            for rank in range(k):
                token = idx2token[topk_indices[rank]]
                prob = topk_probs[rank]
                print(f"   {rank+1}. {token:<10} | prob = {prob:.4f}")

            next_token_id = topk_indices[0].item()
            tgt = torch.cat([tgt, torch.tensor([[next_token_id]], device=device)], dim=1)

            if idx2token[next_token_id] == config.eos_token:
                break

    # Trả về chuỗi LaTeX được dự đoán
    token_ids = tgt.squeeze().cpu().tolist()
    tokens = [idx2token[idx] for idx in token_ids if idx2token[idx] not in {config.sos_token, config.eos_token, config.pad_token}]
    return ' '.join(tokens)


# === Chạy thử ===
if __name__ == "__main__":
    image_tensor = preprocess_image(image_test_path)
    tokens = predict(image_tensor)
    latex_output = ' '.join(tokens)
    print("Predicted LaTeX:", latex_output)
