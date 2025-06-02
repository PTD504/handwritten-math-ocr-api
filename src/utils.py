import os
import torch
from config import config
import torch.nn as nn
from data_loader import create_vocab

def save_checkpoint(epoch, model, optimizer, loss, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, os.path.join(config.checkpoint_dir, filename))

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, filename))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

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