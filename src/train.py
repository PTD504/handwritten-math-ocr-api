import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import FormulaRecognitionModel
from utils import save_checkpoint, load_checkpoint
from config import config

def train_model(train_loader, val_loader, vocab, device):
    model = FormulaRecognitionModel(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[config.pad_token])
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        for images, captions, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            captions = captions.to(device)
            
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, captions, _ in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(epoch+1, model, optimizer, val_loss, f"checkpoint_epoch_{epoch+1}.pth")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch+1, model, optimizer, val_loss, "best_model.pth")
    
    return model

def load_and_continue_training(train_loader, val_loader, vocab, device, checkpoint_path):
    model = FormulaRecognitionModel(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[config.pad_token])
    
    # Load checkpoint
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Continue training from saved checkpoint
    for epoch in range(start_epoch, config.epochs):
        # Training
        model.train()
        train_loss = 0
        for images, captions, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            captions = captions.to(device)
            
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, captions, _ in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(epoch+1, model, optimizer, val_loss, f"checkpoint_epoch_{epoch+1}.pth")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch+1, model, optimizer, val_loss, "best_model.pth")
    
    return model
    