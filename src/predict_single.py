import torch
import cv2
import logging
from pathlib import Path
from torchvision import transforms
from config import config
from model import FormulaRecognitionModel
from utils import load_checkpoint, create_vocab_dicts
from inference import predict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ImagePredictor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vocab, self.idx2char = create_vocab_dicts()
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def _load_model(self):
        """Load trained model from checkpoint"""
        model = FormulaRecognitionModel(len(self.vocab)).to(self.device)
        try:
            checkpoint_path = Path(config.checkpoint_dir) / "best_model.pth"
            load_checkpoint(model, None, checkpoint_path)
            model.eval()
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def preprocess_image(self, image_path):
        """Load and preprocess single image"""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        # Resize with padding preservation
        img = cv2.resize(img, (config.img_w, config.img_h))
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict_image(self, image_path):
        """Predict formula from single image"""
        try:
            # Preprocess
            image_tensor = self.preprocess_image(image_path)
            
            # Predict
            with torch.no_grad():
                formula = predict(image_tensor[0], self.model, self.vocab, self.idx2char, self.device)
            
            logger.info(f"Predicted formula: {formula}")
            return {
                'image_path': str(image_path),
                'prediction': formula,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {str(e)}")
            return {
                'image_path': str(image_path),
                'prediction': None,
                'status': 'failed',
                'error': str(e)
            }

def main():
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Predict math formula from handwritten image')
    parser.add_argument('image_path', type=str, default=r"E:\SchoolWork\CS338\final-project\collect-data\HandwrittenMathematicalExpression\HandwrittenMathematicalExpression\1.png", help='Path to input image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = ImagePredictor(device=args.device)
        
        # Run prediction
        result = predictor.predict_image(args.image_path)
        
        if result['status'] == 'success':
            print("\nPREDICTION RESULT:")
            print(f"Image: {result['image_path']}")
            print(f"Formula: {result['prediction']}")
        else:
            print("\nPREDICTION FAILED:")
            print(f"Error: {result['error']}")
            
    except Exception as e:
        logger.exception("Prediction process failed")
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()