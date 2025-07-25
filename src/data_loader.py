import os
import cv2
from PIL import Image
import torch
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import tokenize_latex
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
        
        img = cv2.resize(img, (config.img_w, config.img_h))
        img = Image.fromarray(img, mode='L')
        img = self.transform(img)
        
        token_ids = [self.vocab[config.sos_token]]
        tokens = tokenize_latex(label)
        token_ids += [self.vocab.get(token, self.vocab[config.unk_token]) for token in tokens]
        token_ids.append(self.vocab[config.eos_token])
        
        length = len(token_ids)
        padded_ids = token_ids[:config.max_seq_len]
        if len(padded_ids) < config.max_seq_len:
            padded_ids += [self.vocab[config.pad_token]] * (config.max_seq_len - len(padded_ids))

        return img, torch.tensor(padded_ids), length

def get_data_loaders(vocab):
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=2, shear=2, scale=(0.95, 1.05)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = MathFormulaDataset(config.train_img_dir, config.train_label_path, vocab, transform)
    val_dataset = MathFormulaDataset(config.val_img_dir, config.val_label_path, vocab)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers if torch.cuda.is_available() else 0,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers if torch.cuda.is_available() else 0,
        persistent_workers=True,
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
        num_workers=config.num_workers if torch.cuda.is_available() else 0,
        persistent_workers=True,
        pin_memory=True
    )

    return test_loader