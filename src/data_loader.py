import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config

class MathFormulaDataset(Dataset):
    def __init__(self, img_dir, label_path, vocab, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(label_path)
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Resize với padding
        img = cv2.resize(img, (config.img_w, config.img_h))
        img = self.transform(img)
        
        # Chuyển label thành token IDs
        token_ids = [self.vocab[config.sos_token]]
        token_ids += [self.vocab.get(char, self.vocab[config.unk_token]) for char in label.split()]
        token_ids.append(self.vocab[config.eos_token])
        
        # Padding sequence
        padded_ids = token_ids[:config.max_seq_len]
        if len(padded_ids) < config.max_seq_len:
            padded_ids += [self.vocab[config.pad_token]] * (config.max_seq_len - len(padded_ids))
            
        return img, torch.tensor(padded_ids), len(token_ids)

def create_vocab(label_paths):
    all_chars = set()
    for path in label_paths:
        df = pd.read_csv(path)
        for formula in df.iloc[:, 1]:
            all_chars.update(formula.split())
    
    vocab = {token: idx for idx, token in enumerate(config.special_tokens + sorted(all_chars))}
    return vocab

def get_data_loaders(vocab):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomAffine(degrees=2, shear=2, scale=(0.95, 1.05))  # Data augmentation
    ])
    
    train_dataset = MathFormulaDataset(config.train_img_dir, config.train_label_path, vocab, transform)
    val_dataset = MathFormulaDataset(config.val_img_dir, config.val_label_path, vocab)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_test_loader(vocab):
    test_dataset = MathFormulaDataset(
        img_dir=config.test_img_dir,
        label_path=config.test_label_path,
        vocab=vocab
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    return test_loader