import torch
from torch.nn import functional as F
from model import FormulaRecognitionModel
from config import config

def predict(images, model, vocab, idx2char, device, mode='greedy', beam_size=3):
    """
    Dự đoán caption cho batch ảnh với greedy hoặc beam search.
    Args:
        images: tensor [B, C, H, W]
        model: mô hình đã load
        vocab: từ điển {token: idx}
        idx2char: từ điển {idx: token}
        device: thiết bị
        mode: 'greedy' hoặc 'beam'
        beam_size: beam width nếu dùng beam search
    Returns:
        List các chuỗi caption dự đoán
    """
    model.eval()
    images = images.to(device)
    batch_size = images.size(0)
    with torch.no_grad():
        encoder_out = model.encoder(images)  # [B, L, D]
        
        if mode == 'greedy':
            ys = torch.full((batch_size, 1), vocab[config.sos_token], dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(config.max_seq_len):
                out = model.decoder(encoder_out, ys)  # [B, T, V]
                next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
                ys = torch.cat([ys, next_token], dim=1)

                finished |= (next_token.squeeze(1) == vocab[config.eos_token])
                if finished.all():
                    break

            sequences = ys  # [B, T]
        
        elif mode == 'beam':
            sequences = []
            for i in range(batch_size):
                enc_out = encoder_out[i].unsqueeze(0)  # [1, L, D]
                beam = [(torch.tensor([vocab[config.sos_token]], device=device), 0.0)]

                for _ in range(config.max_seq_len):
                    new_beam = []
                    for seq, score in beam:
                        if seq[-1].item() == vocab[config.eos_token]:
                            new_beam.append((seq, score))
                            continue
                        out = model.decoder(enc_out, seq.unsqueeze(0))  # [1, T, V]
                        next_token_logits = out[0, -1]  # [V]
                        log_probs = F.log_softmax(next_token_logits, dim=-1)
                        topk_log_probs, topk_indices = log_probs.topk(beam_size)

                        for j in range(beam_size):
                            next_seq = torch.cat([seq, topk_indices[j].unsqueeze(0)])
                            new_score = score + topk_log_probs[j].item()
                            new_beam.append((next_seq, new_score))

                    new_beam.sort(key=lambda x: x[1], reverse=True)
                    beam = new_beam[:beam_size]
                    if all(s[-1].item() == vocab[config.eos_token] for s, _ in beam):
                        break

                best_seq = beam[0][0]
                sequences.append(best_seq)

            # Padding sequences to same length
            max_len = max(seq.size(0) for seq in sequences)
            sequences = torch.stack([
                torch.cat([seq, torch.full((max_len - seq.size(0),), vocab[config.pad_token], device=device)])
                for seq in sequences
            ], dim=0)  # [B, T]

        else:
            raise ValueError("Mode must be either 'greedy' or 'beam'.")

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