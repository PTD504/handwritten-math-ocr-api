import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent  # app/
    MODEL_DIR = BASE_DIR / 'trained-model'
    SRC_DIR = BASE_DIR / 'src'
    
    # Legacy support (fix typo in original)
    @property
    def model_dir(self):
        return str(self.MODEL_DIR)
    
    # API Configuration
    API_TITLE = "Handwritten Math Formula Recognition API"
    API_DESCRIPTION = "Convert handwritten mathematical formulas to LaTeX using deep learning"
    API_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8080
    
    # Model parameters
    IMG_H = 96
    IMG_W = 320
    D_MODEL = 256
    NHEAD = 8
    NUM_DECODER_LAYERS = 8
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.2
    MAX_SEQ_LEN = 150
    
    # Legacy support for lowercase
    img_h = IMG_H
    img_w = IMG_W
    d_model = D_MODEL
    nhead = NHEAD
    num_decoder_layers = NUM_DECODER_LAYERS
    dim_feedforward = DIM_FEEDFORWARD
    dropout = DROPOUT
    max_seq_len = MAX_SEQ_LEN
    
    # Vocabulary settings
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    
    # Legacy support
    sos_token = SOS_TOKEN
    eos_token = EOS_TOKEN
    pad_token = PAD_TOKEN
    unk_token = UNK_TOKEN
    special_tokens = SPECIAL_TOKENS
    
    # Prediction settings
    DEFAULT_BEAM_SIZE = 3
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Image preprocessing
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Error messages
    ERROR_MESSAGES = {
        'file_too_large': 'File size exceeds the maximum limit of 10MB.',
        'invalid_format': 'Invalid file format. Please upload an image file.',
        'processing_error': 'An error occurred while processing the image. Please try again.',
        'model_not_loaded': 'Model is not properly loaded. Please contact support.',
        'empty_result': 'Could not extract formula from the image. Please try with a clearer image.'
    }

config = Config()