import os
import torch
from config import config
import torch.nn as nn
import json
from data_loader import create_vocab

def save_checkpoint(epoch, model, optimizer, scaler, scheduler, accuracy, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),  
        'scheduler_state_dict': scheduler.state_dict(),
        'metric_value': accuracy,
    }
    torch.save(state, os.path.join(config.checkpoint_dir, filename))


def load_checkpoint(model, optimizer, scaler=None, scheduler=None, filename='best_model.pth'):
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, filename))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
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

def create_vocab_dicts():
    vocab = create_vocab([config.train_label_path])
    idx2char = {idx: char for char, idx in vocab.items()}
    return vocab, idx2char

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

def get_vocab_size(vocab):
    return len(vocab)
