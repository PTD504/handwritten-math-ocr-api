import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from model_res18trans import FormulaRecognitionModel
# from model_swin import FormulaRecognitionModel
from config import config
from utils import save_checkpoint, load_checkpoint, compute_metrics
import os

def train_model(train_loader, val_loader, vocab, tokenizer, device, patience=5):
    # Check if this is a training process from starting or from checkpoint
    if config.start_training:
        config.epochs = 15
        
    model = FormulaRecognitionModel(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[config.pad_token], label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_edit_dist = float('inf')
    no_improvement_epochs = 0

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    train_loss_history, val_loss_history, cer_history = [], [], []
    bleu_score_history, editdist_history = [], []

    # Start MLflow tracking
    with mlflow.start_run():
        # Log config hyperparameters
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("epochs", config.epochs)
        mlflow.log_param("model architecture", "ENCODER: ResNet18 + Transformer - DECODER: Transformer")
        mlflow.log_param("dropout", config.dropout)
        mlflow.log_param("max_seq_len", config.max_seq_len)
        mlflow.log_param("vocab_size", len(vocab))
        mlflow.log_param("train_size", len(train_loader.dataset))
        mlflow.log_param("val_size", len(val_loader.dataset))
        mlflow.log_param("total_params_M", round(total_params / 1e6, 2))

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

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch+1)
            mlflow.log_metric("val_loss", val_loss, step=epoch+1)
            mlflow.log_metric("edit_distance", metrics['edit_distance'], step=epoch+1)
            mlflow.log_metric("cer", metrics['cer'], step=epoch+1)
            mlflow.log_metric("bleu", metrics['bleu'], step=epoch+1)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            cer_history.append(metrics["cer"])
            bleu_score_history.append(metrics["bleu"])
            editdist_history.append(metrics["edit_distance"])

            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, metrics['edit_distance'], ckpt_path)
                mlflow.log_artifact(ckpt_path)

            if metrics['edit_distance'] < best_val_edit_dist:
                best_val_edit_dist = metrics['edit_distance']
                no_improvement_epochs = 0
                save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, metrics['edit_distance'], "best_model.pth")
                mlflow.pytorch.log_model(model, "best_model")
                print(f"New best model saved with edit distance: {best_val_edit_dist:.2f}")
            else:
                no_improvement_epochs += 1
                print(f"No improvement for {no_improvement_epochs}/{patience} epochs")

            if no_improvement_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        cer_normalized = [(1 - cer) for cer in cer_history]

        # Plot and save curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(bleu_score_history, label="BLEU score")
        plt.plot(editdist_history, label="Levenshtein distance")
        plt.plot(cer_normalized, label="Inversed CER (1 - CER)")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"training_curves_{config.epochs}.png")
        mlflow.log_artifact(f"training_curves_{config.epochs}.png")
        plt.close()

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

    train_loss_history, val_loss_history, cer_history = [], [], []
    bleu_score_history, editdist_history = [], []

    with mlflow.start_run(id="123"):

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

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch+1)
            mlflow.log_metric("val_loss", val_loss, step=epoch+1)
            mlflow.log_metric("edit_distance", metrics['edit_distance'], step=epoch+1)
            mlflow.log_metric("cer", metrics['cer'], step=epoch+1)
            mlflow.log_metric("bleu", metrics['bleu'], step=epoch+1)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            cer_history.append(metrics["cer"])
            bleu_score_history.append(metrics["bleu"])
            editdist_history.append(metrics["edit_distance"])

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
        
        cer_normalized = [(1 - cer) for cer in cer_history]

        if config.epcohs < 30:
            epochs = list(range(1, config.epochs + 1))
        else:
            epochs = list(range(15, config.epochs + 1))

        # Plot and save curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_history, label="Train Loss")
        plt.plot(epochs, val_loss_history, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, bleu_score_history, label="BLEU score")
        plt.plot(epochs, editdist_history, label="Levenshtein distance")
        plt.plot(epochs, cer_normalized, label="Inversed CER (1 - CER)")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"training_curves_{config.epochs}.png")
        mlflow.log_artifact(f"training_curves_{config.epochs}.png")
        plt.close()

    return model