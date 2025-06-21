import os

class Config:
    model_dir = os.path.join(os.path.dirname(os.getcwd()), 'api', 'trained-model')
    
    # Model parameters
    img_h = 96
    img_w = 320
    d_model = 256
    nhead = 8
    num_decoder_layers = 8
    dim_feedforward = 512
    dropout = 0.2

    max_seq_len = 150
    
    # Vocabulary settings
    sos_token = '<sos>'
    eos_token = '<eos>'
    pad_token = '<pad>'
    unk_token = '<unk>'
    special_tokens = [pad_token, sos_token, eos_token, unk_token]

config = Config()