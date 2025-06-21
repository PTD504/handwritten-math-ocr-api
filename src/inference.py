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
        
        if mode == 'greedy':
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
        
        elif mode == 'beam':
            sequences = beam_search_batch(encoder_out, model, vocab, device, beam_size)
        
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

def beam_search_batch(encoder_out, model, vocab, device, beam_size):
    batch_size, enc_len, enc_dim = encoder_out.shape
    
    beams = torch.full((batch_size * beam_size, 1), vocab[config.sos_token], 
                      dtype=torch.long, device=device)
    
    beam_scores = torch.zeros(batch_size * beam_size, device=device)
    
    encoder_out_expanded = encoder_out.unsqueeze(1).expand(-1, beam_size, -1, -1)
    encoder_out_expanded = encoder_out_expanded.contiguous().view(-1, enc_len, enc_dim)
    
    finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)
    
    for step in range(config.max_seq_len):
        if finished.all():
            break
            
        decoder_out = model.decoder(encoder_out_expanded, beams)
        next_token_logits = decoder_out[:, -1, :]
        
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        vocab_size = log_probs.size(-1)
        if step == 0:
            scores = log_probs.view(batch_size, beam_size, vocab_size)[:, 0, :]
        else:
            scores = beam_scores.unsqueeze(-1) + log_probs
            scores = scores.view(batch_size, beam_size * vocab_size)
        
        top_scores, top_indices = scores.topk(beam_size, dim=-1)
        
        if step == 0:
            beam_indices = torch.zeros_like(top_indices)
            token_indices = top_indices
        else:
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

        new_beams = []
        new_scores = []
        new_finished = []
        
        for b in range(batch_size):
            for k in range(beam_size):
                if step == 0:
                    orig_beam_idx = b * beam_size
                else:
                    orig_beam_idx = b * beam_size + beam_indices[b, k]
                
                old_seq = beams[orig_beam_idx]
                
                new_token = token_indices[b, k].unsqueeze(0)
                new_seq = torch.cat([old_seq, new_token])
                
                new_beams.append(new_seq)
                new_scores.append(top_scores[b, k])
                
                is_finished = (new_token.item() == vocab[config.eos_token]) or finished[orig_beam_idx]
                new_finished.append(is_finished)
        
        max_len = max(seq.size(0) for seq in new_beams)
        beams = torch.stack([
            F.pad(seq, (0, max_len - seq.size(0)), value=vocab[config.pad_token])
            for seq in new_beams
        ])
        
        beam_scores = torch.tensor(new_scores, device=device)
        finished = torch.tensor(new_finished, device=device)
    
    sequences = []
    for b in range(batch_size):
        batch_beams = beams[b * beam_size:(b + 1) * beam_size]
        batch_scores = beam_scores[b * beam_size:(b + 1) * beam_size]
        
        best_idx = batch_scores.argmax()
        best_seq = batch_beams[best_idx]
        
        sequences.append(best_seq)
    
    max_len = max(seq.size(0) for seq in sequences)
    sequences = torch.stack([
        F.pad(seq, (0, max_len - seq.size(0)), value=vocab[config.pad_token])
        for seq in sequences
    ])
    
    return sequences
