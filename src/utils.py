import os
import torch
from config import config
import torch.nn as nn
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
import editdistance

def compute_metrics(pred_ids_list, tgt_ids_list, tokenizer, eos_token, pad_token):
    # Compute all evaluation metrics: Edit Distance, CER and BLEU
    pred_strs = [tokenizer.decode(ids, eos_token=eos_token, pad_token=pad_token) for ids in pred_ids_list]
    tgt_strs = [tokenizer.decode(ids, eos_token=eos_token, pad_token=pad_token) for ids in tgt_ids_list]
    
    # Compute edit distance (Levenshtein distance)
    edit_distances = [
        editdistance.eval(pred, tgt)
        for pred, tgt in zip(pred_strs, tgt_strs)
    ]
    avg_edit_distance = np.mean(edit_distances)
    
    # Compute CER (Character Error Rate)
    total_chars = sum(len(tgt) for tgt in tgt_strs)
    total_errors = sum(editdistance.eval(pred, tgt) for pred, tgt in zip(pred_strs, tgt_strs))
    avg_cer = total_errors / total_chars if total_chars > 0 else 0
    
    # Compute BLEU score
    bleu_score = compute_bleu_score(pred_ids_list, tgt_ids_list, tokenizer, eos_token, pad_token)
    
    return {
        'edit_distance': avg_edit_distance,
        'cer': avg_cer,
        'bleu': bleu_score
    }

def compute_bleu_score(pred_ids_list, tgt_ids_list, tokenizer, eos_token, pad_token):
    references = []
    hypotheses = []

    for tgt_ids, pred_ids in zip(tgt_ids_list, pred_ids_list):
        ref_str = tokenizer.decode(tgt_ids, eos_token=eos_token, pad_token=pad_token)
        hyp_str = tokenizer.decode(pred_ids, eos_token=eos_token, pad_token=pad_token)
        
        ref_tokens = ref_str.split()
        hyp_tokens = hyp_str.split()
        
        references.append([ref_tokens])  
        hypotheses.append(hyp_tokens)

    smoothie = SmoothingFunction().method4

    bleu_score = corpus_bleu(
        references,
        hypotheses,
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25)
    )

    return bleu_score

def save_checkpoint(epoch, model, optimizer, scaler, scheduler, metric_value, filename):
    # Save training checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metric_value': metric_value
    }
    torch.save(checkpoint, os.path.join(config.checkpoint_dir, filename))

def load_checkpoint(model, optimizer, scaler, scheduler, filename):
    # Load training checkpoint
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, filename), map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['metric_value']

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) in [nn.LSTM, nn.GRU, nn.RNN]:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

def tokenize_latex(formula: str):
    token_pattern = r'(\\[a-zA-Z]+|[{}_^$%&#]|[0-9]+|[a-zA-Z]+|[^\s])'
    tokens = re.findall(token_pattern, formula)
    return tokens

def create_vocab(label_paths):
    all_tokens = set()

    for path in label_paths:
        df = pd.read_csv(path)
        for formula in df['latex_label'].dropna():
            formula = formula.strip()
            tokens = tokenize_latex(formula)
            all_tokens.update(tokens)

    vocab = {token: idx for idx, token in enumerate(config.special_tokens + sorted(all_tokens))}
    return vocab

def save_vocab(vocab, filename='vocab.json'):
    data = {
        'vocab': vocab,
        'idx2char': {idx: char for char, idx in vocab.items()}
    }
    with open(os.path.join(config.checkpoint_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_vocab(filename='vocab.json'):
    with open(os.path.join(config.checkpoint_dir, filename), 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = data['vocab']
    idx2char = {int(k): v for k, v in data['idx2char'].items()}
    return vocab, idx2char
