import torch
import torch.nn.functional as F
import heapq
from utils import load_vocab, tokens_to_latex
from config import config
from model import FormulaRecognitionModel
import os

def load_model(model_path: str, vocab, device):
    """
    Tải mô hình đã huấn luyện, bao gồm mô hình, optimizer, scaler và scheduler từ checkpoint.
    
    Args:
        model_path (str): Đường dẫn đến file mô hình (.pth).
        device (torch.device): Thiết bị để tải mô hình (CPU hoặc GPU).
    
    Returns:
        model (torch.nn.Module): Mô hình đã được tải.
        optimizer (torch.optim.Optimizer): Optimizer đã được tải.
        scaler (torch.cuda.amp.GradScaler): Scaler đã được tải.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Scheduler đã được tải.
        epoch (int): Epoch bắt đầu khi tiếp tục huấn luyện.
    """
    # Tạo mô hình
    model = FormulaRecognitionModel(len(vocab)).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model


def predict(model, image_tensor: torch.Tensor, vocab: dict, idx2char: dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"DEBUG (predict): Image tensor shape: {image_tensor.shape}")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    batch_size = 1
    target = torch.tensor([[vocab[config.sos_token]]], dtype=torch.long).to(device)
    print(f"DEBUG (predict): Initial target shape: {target.shape}")

    output_tokens = []
    for i in range(config.max_seq_len): # Thêm biến lặp `i`
        print(f"DEBUG (predict): Decoding step {i+1}/{config.max_seq_len}, current target shape: {target.shape}")
        with torch.no_grad():
            output = model(image_tensor, target)
            print(f"DEBUG (predict): Model output shape: {output.shape}")

            next_token_logits = output[:, -1, :]
            print(f"DEBUG (predict): Next token logits shape: {next_token_logits.shape}")

            # Kiểm tra next_token_logits trước khi argmax
            if next_token_logits.numel() == 0:
                print("DEBUG (predict): ERROR: next_token_logits is empty! This should not happen if output is valid.")
                # Điều này rất khó xảy ra nếu output có shape [1, seq_len, vocab_size] và seq_len > 0.
                # Nếu nó xảy ra, nghĩa là output[:, -1, :] đã trả về một tensor rỗng.
                # Kiểm tra lại model()
                break # Thoát vòng lặp để tránh lỗi

            next_token = torch.argmax(F.softmax(next_token_logits, dim=-1), dim=-1)
            print(f"DEBUG (predict): Predicted next_token ID: {next_token.item()}")

            if next_token.item() == vocab[config.eos_token]:
                print(f"DEBUG (predict): End of sequence token <eos> detected.")
                break

            output_tokens.append(next_token.item())
            print(f"DEBUG (predict): Appended token: {next_token.item()}. Current output_tokens length: {len(output_tokens)}")

            target = torch.cat([target, next_token.unsqueeze(-1)], dim=-1)

    print(f"DEBUG (predict): Final output_tokens list: {output_tokens}")
    if not output_tokens:
        print("DEBUG (predict): WARNING: output_tokens list is EMPTY before calling tokens_to_latex!")
        return r"\text{Không thể dự đoán công thức từ ảnh.} Kiểm tra lại mô hình."

    formula = tokens_to_latex(output_tokens, idx2char)
    print(f"DEBUG (predict): Final predicted formula: {formula}")

    return formula
