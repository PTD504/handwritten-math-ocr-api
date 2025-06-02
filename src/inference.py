import torch
from torch.nn import functional as F
from model import FormulaRecognitionModel
from config import config

def predict(image, model, vocab, idx2char, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        features = model.encoder(image)
        
        # Beam search
        current_beam = [(torch.tensor([vocab[config.sos_token]]).to(device), 0.0)]
        
        for _ in range(config.max_seq_len - 1):
            new_beam = []
            for seq, score in current_beam:
                if seq[-1] == vocab[config.eos_token]:
                    new_beam.append((seq, score))
                    continue
                
                output = model.decoder(features, seq.unsqueeze(0))
                next_token_probs = F.log_softmax(output[0, -1], dim=-1)
                topk_probs, topk_tokens = next_token_probs.topk(config.beam_size)
                
                for i in range(config.beam_size):
                    new_seq = torch.cat([seq, topk_tokens[i].unsqueeze(0)])
                    new_score = score + topk_probs[i].item()
                    new_beam.append((new_seq, new_score))
            
            # Choose top beam_size sequences
            new_beam.sort(key=lambda x: x[1], reverse=True)
            current_beam = new_beam[:config.beam_size]
            
            # Check if all sequences in the beam have reached <eos>
            if all(seq[-1] == vocab[config.eos_token] for seq, _ in current_beam):
                break
        
        best_seq = current_beam[0][0]
        tokens = [idx2char[idx.item()] for idx in best_seq if idx != vocab[config.pad_token]]
        return ' '.join(tokens[1:-1])  # Remove <sos> and <eos>