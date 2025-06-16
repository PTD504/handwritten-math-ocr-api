import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import FormulaRecognitionModel
from utils import save_checkpoint, load_checkpoint
from config import config
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(train_loader, val_loader, vocab, device, patience=5, epochs=40):
    config.epochs = epochs
    
    model = FormulaRecognitionModel(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[config.pad_token])
    scaler = torch.amp.GradScaler('cuda')  # Mixed precision training
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = 0.0
    no_improvement_epochs = 0  # Counter for early stopping

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        for images, captions, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            # Mixed precision training
            with torch.amp.autocast('cuda'):
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
            
            optimizer.zero_grad(set_to_none=True)  # Set gradients to None for efficiency
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # Validation
        val_loss = 0
        val_accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, captions, _ in val_loader:
                images = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)   
                
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
                val_loss += loss.item()

                # Calculate validation accuracy
                pred = outputs.argmax(dim=-1)
                correct = (pred == captions[:, 1:]).sum().item()
                total = captions[:, 1:].numel()
                val_accuracy += correct / total

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)  # Average over batches

        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        # Step the scheduler
        scheduler.step(val_loss)

        for param_group in optimizer.param_groups:
            print(f"Learning Rate after epoch {epoch+1}: {param_group['lr']:.6f}")
        
        if (epoch + 1) % 5 == 0: 
            save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, val_accuracy, f"checkpoint_epoch_{epoch+1}.pth")
        
        # Check for best model based on val_accuracy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0  # Reset counter
            save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, val_loss, "best_model.pth")
        else:
            no_improvement_epochs += 1

        # Early Stopping: Stop training if no improvement in `patience` epochs
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
    
    return model

def load_and_continue_training(train_loader, val_loader, vocab, device, checkpoint_filename="best_model.pth", patience=5):
    model = FormulaRecognitionModel(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[config.pad_token])
    scaler = torch.amp.GradScaler('cuda')  # Mixed precision training
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Load checkpoint
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scaler, scheduler, checkpoint_filename)
    no_improvement_epochs = 0  # Counter for early stopping

    # Continue training from saved checkpoint
    for epoch in range(start_epoch, config.epochs):
        # Training
        model.train()
        train_loss = 0
        for images, captions, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            # Mixed precision training
            with torch.amp.autocast('cuda'):
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
            
            optimizer.zero_grad(set_to_none=True)  # Set gradients to None for efficiency
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        val_loss = 0
        val_accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, captions, _ in val_loader:
                images = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)
                
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
                val_loss += loss.item()

                # Calculate validation accuracy
                pred = outputs.argmax(dim=-1)
                correct = (pred == captions[:, 1:]).sum().item()
                total = captions[:, 1:].numel()
                val_accuracy += correct / total
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)  # Average over batches

        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        # Step the scheduler
        scheduler.step(val_loss)

        for param_group in optimizer.param_groups:
            print(f"Learning Rate after epoch {epoch+1}: {param_group['lr']:.6f}")
        
        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, val_accuracy, f"checkpoint_epoch_{epoch+1}.pth")
        
        # Check for best model based on val_accuracy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0  # Reset counter
            save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, val_loss, "best_model.pth")
        else:
            no_improvement_epochs += 1

        # Early Stopping: Stop training if no improvement in `patience` epochs
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
    
    return model
    