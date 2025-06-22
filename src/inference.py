import torch
from torch.nn import functional as F
from model import FormulaRecognitionModel
from config import config

def predict(images, model, vocab, idx2char, device, mode='greedy', beam_size=3):
    model.eval()
    images = images.to(device)
    batch_size = images.size(0)
    
    with torch.no_grad():
        encoder_out = model.encoder(images)
        
        ys = torch.full((batch_size, 1), vocab[config.sos_token], dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(config.max_seq_len):
            out = model.decoder(encoder_out, ys) 
            next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)  
            ys = torch.cat([ys, next_token], dim=1)

            finished |= (next_token.squeeze(1) == vocab[config.eos_token])
            if finished.all():
                break

        sequences = ys 

        # Decode sequences to string
        results = []
        for seq in sequences:
            tokens = []
            for idx in seq:
                token = idx2char[idx.item()]
                if token in [config.sos_token, config.pad_token]:
                    continue
                if token == config.eos_token:
                    break
                tokens.append(token)
            results.append(' '.join(tokens))

        return results