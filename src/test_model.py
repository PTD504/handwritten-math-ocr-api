import torch
import pandas as pd
import logging
from tqdm.auto import tqdm
import os
from pathlib import Path
from config import config
from data_loader import get_test_loader
from model import FormulaRecognitionModel
from utils import load_checkpoint, create_vocab_dicts
from inference import predict

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Initialize hardware settings"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    return device

def load_model(vocab_size, device):
    """Load trained model from checkpoint"""
    model = FormulaRecognitionModel(vocab_size).to(device)
    try:
        # checkpoint_path = Path(config.checkpoint_dir) / "best_model.pth"
        checkpoint_path = Path(config.checkpoint_dir) / "checkpoint_epoch_27.pth"
        # load_checkpoint(model, None, checkpoint_path)
        load_checkpoint(model, optimizer=None, scaler=None, scheduler=None, filename=checkpoint_path)
        
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
            
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

def calculate_metrics(pred, truth):
    """Calculate accuracy and CER using optimized libraries"""
    try:
        import Levenshtein
        cer = Levenshtein.distance(pred, truth) / max(len(truth), 1)
    except ImportError:
        logger.warning("Levenshtein package not found, using difflib")
        from difflib import SequenceMatcher
        cer = 1 - SequenceMatcher(None, pred, truth).ratio()
    
    is_correct = pred == truth
    return is_correct, cer

def evaluate_model(model, test_loader, vocab, idx2char, device, mode='greedy'):
    """Evaluate the model on the test set and return results as a DataFrame"""
    model.eval()
    results = []

    with torch.no_grad():
        for images, captions, lengths in tqdm(test_loader, desc=f"Evaluating ({mode})"):
            images = images.to(device)
            captions = captions.to(device)

            predictions = predict(images, model, vocab, idx2char, device, mode=mode)

            for i in range(len(predictions)):
                pred = predictions[i]
                gt_tokens = [idx2char[idx.item()] for idx in captions[i][1:lengths[i].item()-1]]
                truth = ' '.join(gt_tokens)

                correct, cer = calculate_metrics(pred, truth)

                results.append({
                    'image_id': test_loader.dataset.df.iloc[i, 0],
                    'prediction': pred,
                    'ground_truth': truth,
                    'is_correct': correct,
                    'cer': cer
                })

    return pd.DataFrame(results)

def save_results(results_df, output_dir="results"):
    """Save evaluation results with proper directory handling"""
    Path(output_dir).mkdir(exist_ok=True)
    result_path = Path(output_dir) / "test_results.csv"
    
    results_df.to_csv(result_path, index=False)
    logger.info(f"Results saved to {result_path}")
    
    # Add summary statistics
    summary = {
        'accuracy': results_df['is_correct'].mean(),
        'avg_cer': results_df['cer'].mean(),
        'total_samples': len(results_df)
    }
    
    with open(Path(output_dir) / "summary.txt", 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    
    return summary

def main():
    try:
        logger.info("Starting evaluation process")
        
        # Setup environment
        device = setup_environment()
        
        # Load data and model
        vocab, idx2char = create_vocab_dicts()
        test_loader = get_test_loader(vocab)
        model = load_model(len(vocab), device)
        
        # Run evaluation
        results_df = evaluate_model(model, test_loader, vocab, idx2char, device)
        
        # Save and display results
        summary = save_results(results_df)
        logger.info(f"\nEvaluation Summary:")
        logger.info(f"Accuracy: {summary['accuracy']:.2%}")
        logger.info(f"Avg CER: {summary['avg_cer']:.4f}")
        logger.info(f"Total Samples: {summary['total_samples']}")
        
    except Exception as e:
        logger.exception("Evaluation failed:")
        raise

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()