import torch
import torch.nn.functional as F
from utils import tokens_to_latex, clean_latex_output
from config import config
from model_swin import FormulaRecognitionModel

def load_model(model_path: str, vocab, device):
    # model = FormulaRecognitionModel(len(vocab)).to(device)
    # checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = torch.load(model_path, map_location=device, weights_only=False)
    
    return model

def predict(model, image_tensor: torch.Tensor, vocab: dict, idx2char: dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)

    target = torch.tensor([[vocab[config.sos_token]]], dtype=torch.long).to(device)
    
    output_tokens = []
    log_probs_sum = 0.0
    
    for i in range(config.max_seq_len):
        with torch.no_grad():
            output = model(image_tensor, target)
            next_token_logits = output[:, -1, :]

            if next_token_logits.numel() == 0:
                break

            probs = F.softmax(next_token_logits, dim=-1)
            
            next_token = torch.argmax(probs, dim=-1)
            
            log_probs = torch.log(probs + 1e-10)
            selected_log_prob = log_probs[0, next_token.item()].item()
            log_probs_sum += selected_log_prob

            if next_token.item() == vocab[config.eos_token]:
                break

            output_tokens.append(next_token.item())
            target = torch.cat([target, next_token.unsqueeze(-1)], dim=-1)

    if not output_tokens:
        return r"\text{Unable to detect a formula from the image. Please verify the model.}", 0.0
    
    average_log_prob = log_probs_sum / len(output_tokens)
    
    formula = tokens_to_latex(output_tokens, idx2char)
    cleaned_formula = clean_latex_output(formula)

    confidence_score = torch.exp(torch.tensor(average_log_prob)).item()
    
    return cleaned_formula, confidence_score