import torch
import torch.nn.functional as F
from utils import tokens_to_latex, clean_latex_output
from config import config
from model import FormulaRecognitionModel

def load_model(model_path: str, len_vocab, device):
    model = FormulaRecognitionModel(len_vocab).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def predict(model, image_tensor: torch.Tensor, vocab: dict, idx2char: dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', beam_size: int = 3):
    model = model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)
    
    initial_sequence = [vocab[config.sos_token]]
    beams = [(initial_sequence, 0.0)]
    
    completed_beams = []
    
    for _ in range(config.max_seq_len):
        new_beams = []
        
        for sequence, log_prob in beams:
            target = torch.tensor([sequence], dtype=torch.long).to(device)
            
            with torch.no_grad():
                output = model(image_tensor, target)
                next_token_logits = output[:, -1, :]
                
                if next_token_logits.numel() == 0:
                    continue
                
                probs = F.softmax(next_token_logits, dim=-1)
                log_probs = torch.log(probs + 1e-10)
                
                top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)
                
                for i in range(beam_size):
                    token_id = top_indices[0, i].item()
                    token_log_prob = top_log_probs[0, i].item()
                    
                    new_sequence = sequence + [token_id]
                    new_log_prob = log_prob + token_log_prob
                    
                    if token_id == vocab[config.eos_token]:
                        completed_beams.append((new_sequence, new_log_prob))
                    else:
                        new_beams.append((new_sequence, new_log_prob))
        
        if completed_beams and len(completed_beams) >= beam_size:
            break
            
        if new_beams:
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
        else:
            break
    
    all_beams = completed_beams + beams
    
    if not all_beams:
        return r"\text{Unable to detect a formula from the image. Please verify the model.}"
    
    best_beam = max(all_beams, key=lambda x: x[1])
    best_sequence = best_beam[0]

    output_tokens = []
    for token in best_sequence[1:]:
        if token == vocab[config.eos_token]:
            break
        output_tokens.append(token)
    
    if not output_tokens:
        return r"\text{Unable to detect a formula from the image. Please verify the model.}"
    else:
        average_log_prob = best_beam[1] / len(output_tokens)
    
    formula = tokens_to_latex(output_tokens, idx2char)
    cleaned_formula = clean_latex_output(formula)

    # Calculate confidence score
    confidence_score = torch.exp(torch.tensor(average_log_prob)).item()
    
    return cleaned_formula, confidence_score