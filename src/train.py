import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import FormulaRecognitionModel
# from model_res18trans import FormulaRecognitionModel
# from model_swin import FormulaRecognitionModel
from utils import save_checkpoint, load_checkpoint, compute_metrics
from config import config
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(train_loader, val_loader, vocab, tokenizer, device, patience=5):
    model = FormulaRecognitionModel(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[config.pad_token], label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_edit_dist = float('inf')
    no_improvement_epochs = 0

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for images, captions, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, captions = images.to(device, non_blocking=True), captions.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, captions, _ in val_loader:
                images, captions = images.to(device, non_blocking=True), captions.to(device, non_blocking=True)
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].reshape(-1))
                val_loss += loss.item()

                all_preds.extend(outputs.argmax(-1).tolist())
                all_targets.extend(captions[:, 1:].tolist())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        metrics = compute_metrics(all_preds, all_targets, tokenizer, config.eos_token, config.pad_token)

        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Metrics - Edit Dist: {metrics['edit_distance']:.2f} | CER: {metrics['cer']:.4f} | BLEU: {metrics['bleu']:.4f}")

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, metrics['edit_distance'],
                            f"checkpoint_epoch_{epoch+1}.pth")

        if metrics['edit_distance'] < best_val_edit_dist:
            best_val_edit_dist = metrics['edit_distance']
            no_improvement_epochs = 0
            save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, metrics['edit_distance'], "best_model.pth")
            print(f"New best model saved with edit distance: {best_val_edit_dist:.2f}")
        else:
            no_improvement_epochs += 1
            print(f"No improvement for {no_improvement_epochs}/{patience} epochs")

        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model

def load_and_continue_training(train_loader, val_loader, vocab, tokenizer, device, 
                               checkpoint_filename="best_model.pth", patience=5):
    model = FormulaRecognitionModel(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[config.pad_token], label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    start_epoch, best_val_edit_dist = load_checkpoint(model, optimizer, scaler, scheduler, checkpoint_filename)
    no_improvement_epochs = 0

    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0
        for images, captions, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, captions = images.to(device, non_blocking=True), captions.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, captions, _ in val_loader:
                images, captions = images.to(device, non_blocking=True), captions.to(device, non_blocking=True)
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].reshape(-1))
                val_loss += loss.item()
                all_preds.extend(outputs.argmax(-1).tolist())
                all_targets.extend(captions[:, 1:].tolist())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        metrics = compute_metrics(all_preds, all_targets, tokenizer, config.eos_token, config.pad_token)

        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Metrics - Edit Dist: {metrics['edit_distance']:.2f} | CER: {metrics['cer']:.4f} | BLEU: {metrics['bleu']:.4f}")

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, metrics['edit_distance'],
                            f"checkpoint_epoch_{epoch+1}.pth")

        if metrics['edit_distance'] < best_val_edit_dist:
            best_val_edit_dist = metrics['edit_distance']
            no_improvement_epochs = 0
            save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, metrics['edit_distance'], "best_model.pth")
            print(f"New best model saved with edit distance: {best_val_edit_dist:.2f}")
        else:
            no_improvement_epochs += 1
            print(f"No improvement for {no_improvement_epochs}/{patience} epochs")

        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model
