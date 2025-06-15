import torch
from tqdm import tqdm
from data_loader import create_vocab, get_data_loaders
# from train import train_model, load_and_continue_training
from train_mlflow import train_model
from utils import load_vocab
from inference import predict
from config import config
import pandas as pd

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create vocab
    vocab, idx2char = load_vocab()
    
    # Create data loader
    train_loader, val_loader = get_data_loaders(vocab)
    
    # Train model
    model = train_model(train_loader, val_loader, vocab, device)

    # Or train model from a checkpoint
    # model = load_and_continue_training(train_loader, val_loader, vocab, device)
    
    # Inference
    test_image = next(iter(val_loader))[0][0]
    prediction = predict(test_image, model, vocab, idx2char, device)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()