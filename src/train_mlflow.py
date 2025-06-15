import mlflow
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
from model import FormulaRecognitionModel
from config import config
from utils import save_checkpoint

def train_model(train_loader, val_loader, vocab, device, patience=5):
    # Initialize MLflow
    mlflow.set_experiment("FormulaRecognition")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("num_epochs", config.epochs)
        mlflow.log_param("patience", patience)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("scheduler", "ReduceLROnPlateau")
        mlflow.log_param("vocab_size", len(vocab))
        mlflow.log_param("model_architecture", "ResNet18 + Transformer")
        mlflow.log_param("train_size", len(train_loader.dataset))
        mlflow.log_param("val_size", len(val_loader.dataset))
        mlflow.log_param("device", str(device))

        # Model and optimizer setup
        model = FormulaRecognitionModel(len(vocab)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab[config.pad_token])
        scaler = torch.amp.GradScaler('cuda')
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_val_bleu = 0.0
        no_improvement_epochs = 0

        loss_history = []
        val_loss_history = []
        val_accuracy_history = []
        val_bleu_history = []

        smooth = SmoothingFunction().method1

        for epoch in range(config.epochs):
            start_time = time.time()
            model.train()
            train_loss = 0
            for images, captions, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

                images = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(images, captions)
                    loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            # Validation
            val_loss = 0
            val_accuracy = 0
            val_bleu_scores = []
            model.eval()
            with torch.no_grad():
                for images, captions, _ in val_loader:
                    images = images.to(device, non_blocking=True)
                    captions = captions.to(device, non_blocking=True)
                    outputs = model(images, captions)
                    loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
                    val_loss += loss.item()

                    # Accuracy
                    pred = outputs.argmax(dim=-1)
                    correct = (pred == captions[:, 1:]).sum().item()
                    total = captions[:, 1:].numel()
                    val_accuracy += correct / total

                    # BLEU
                    for i in range(captions.size(0)):
                        pred_seq = pred[i].cpu().numpy().tolist()
                        true_seq = captions[i, 1:].cpu().tolist()
                        pred_seq = [p for p in pred_seq if p != vocab[config.pad_token]]
                        true_seq = [t for t in true_seq if t != vocab[config.pad_token]]
                        if len(pred_seq) > 0 and len(true_seq) > 0:
                            score = sentence_bleu([true_seq], pred_seq, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
                            val_bleu_scores.append(score)

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy /= len(val_loader)
            val_bleu = np.mean(val_bleu_scores) if val_bleu_scores else 0.0

            # Update learning rate
            scheduler.step(val_bleu)

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_bleu", val_bleu, step=epoch)

            # Save history
            loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)
            val_bleu_history.append(val_bleu)

            # Print log
            epoch_time = time.time() - start_time
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)

            save_checkpoint(epoch+1, model, optimizer, scaler, scheduler, val_accuracy, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val BLEU: {val_bleu:.4f}")

            # Save checkpoint
            mlflow.log_artifact(f"checkpoint_{epoch+1}.pth")
            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                mlflow.log_artifact("best_model.pth")
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        # Plot and save curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracy_history, label="Val Accuracy")
        plt.plot(val_bleu_history, label="Val BLEU")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_curves.png")
        mlflow.log_artifact("training_curves.png")
        plt.close()

    return model
