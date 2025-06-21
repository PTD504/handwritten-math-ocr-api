from utils import create_vocab, save_vocab
from config import config

def main():
    vocab = create_vocab([config.train_label_path])
    save_vocab(vocab)
    print(f"Saved vocab with {len(vocab)} tokens to {config.checkpoint_dir}/vocab.json")

if __name__ == "__main__":
    main()
