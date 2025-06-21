import torch
from tqdm import tqdm
from data_loader import create_vocab, get_data_loaders
# from train_mlflow import train_model, load_and_continue_training
from train import train_model, load_and_continue_training
from utils import load_vocab
from inference import predict
from tokenizer import Tokenizer
import pandas as pd

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create vocab
    vocab, idx2char = load_vocab()

    tokenizer = Tokenizer(idx2char)
    
    # Create data loader
    train_loader, val_loader = get_data_loaders(vocab)
    
    # Train model
    model = train_model(train_loader=train_loader, val_loader=val_loader, vocab=vocab, tokenizer=tokenizer, device=device)

    # Or train model from a checkpoint
    # model = load_and_continue_training(train_loader=train_loader, val_loader=val_loader, vocab=vocab, device=device)
    
    # Inference
    test_image = next(iter(val_loader))[0][0]
    test_image = test_image.unsqueeze(0)

    prediction = predict(test_image, model, vocab, idx2char, device, model='beam')
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()