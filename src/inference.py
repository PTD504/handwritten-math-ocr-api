import torch
from torch.nn import functional as F
from model import FormulaRecognitionModel
from config import config

def predict(image, model, vocab, idx2char, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        features = model.encoder(image)

        current_beam = [(torch.tensor([vocab[config.sos_token]], device=device), 0.0)]

        for _ in range(config.max_seq_len):
            new_beam = []
            for seq, score in current_beam:
                if seq[-1].item() == vocab[config.eos_token]:
                    new_beam.append((seq, score))
                    continue

                output = model.decoder(features, seq.unsqueeze(0))  # [1, T, V]
                next_token_logits = output[0, -1]  # [V]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_indices = log_probs.topk(config.beam_size)

                for i in range(config.beam_size):
                    new_seq = torch.cat([seq, topk_indices[i].unsqueeze(0)])
                    new_score = score + topk_log_probs[i].item()
                    new_beam.append((new_seq, new_score))

            new_beam.sort(key=lambda x: x[1], reverse=True)
            current_beam = new_beam[:config.beam_size]

            if all(seq[-1].item() == vocab[config.eos_token] for seq, _ in current_beam):
                break

        # Final decode best sequence
        best_seq = current_beam[0][0]
        tokens = []
        for idx in best_seq:
            token = idx2char[idx.item()]
            if token == config.sos_token or token == config.pad_token:
                continue
            if token == config.eos_token:
                break
            tokens.append(token)
        return ' '.join(tokens)