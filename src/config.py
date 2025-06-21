import os

class Config:
    # Path settings
    data_root = os.path.join(os.path.dirname(os.getcwd()), 'data')
    train_img_dir = os.path.join(data_root, 'train_formulas')
    val_img_dir = os.path.join(data_root, 'validate_formulas')
    train_label_path = os.path.join(data_root, 'train_labels.csv')
    val_label_path = os.path.join(data_root, 'validate_labels.csv')
    test_img_dir = os.path.join(data_root, 'test_formulas')
    test_label_path = os.path.join(data_root, 'test_labels.csv')
    
    checkpoint_dir = os.path.join(os.path.dirname(os.getcwd()), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Model parameters
    img_w = 320
    img_h = 96
    d_model = 256
    nhead = 8
    dim_feedforward = 512
    dropout = 0.2

    # Encoder: ResNet18 only
    res18_num_decoder_layers = 8

    # Encoder: ResNet18 + Transformer encoder layer
    res18trans_num_encoder_layers = 8
    res18trans_num_decoder_layers = 8

    # Encoder: Swin Transformer
    swin_num_decoder_layers = 8
    
    # Training parameters
    batch_size = 64

    num_workers = 4
    learning_rate = 3e-4
    epochs = 15
    max_seq_len = 150
    
    # Vocabulary settings
    sos_token = '<sos>'
    eos_token = '<eos>'
    pad_token = '<pad>'
    unk_token = '<unk>'
    special_tokens = [pad_token, sos_token, eos_token, unk_token]
    
    # Inference
    beam_size = 5

config = Config()