import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from config import config

class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adaptive pooling to get variable sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.projection = nn.Linear(512, config.d_model)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        return x.squeeze(2).permute(0, 2, 1)  # (batch, seq_len, d_model)

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoder = nn.Embedding(config.max_seq_len, config.d_model)
        
        decoder_layer = TransformerDecoderLayer(
            config.d_model, config.nhead, config.dim_feedforward, config.dropout
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, config.num_decoder_layers)
        self.fc_out = nn.Linear(config.d_model, vocab_size)
        
        # Mask for the target sequence
        self.register_buffer("tgt_mask", self.generate_square_subsequent_mask(config.max_seq_len))

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, encoder_out, tgt):
        # Embedding + positional encoding
        tgt_embed = self.embedding(tgt)
        positions = torch.arange(0, tgt.size(1)).unsqueeze(0).to(tgt.device)
        tgt_embed = tgt_embed + self.pos_encoder(positions)
        
        # Transformer decoder
        tgt_embed = tgt_embed.permute(1, 0, 2)
        output = self.transformer_decoder(
            tgt_embed, 
            encoder_out.permute(1, 0, 2),
            tgt_mask=self.tgt_mask[:tgt.size(1), :tgt.size(1)]
        )
        output = self.fc_out(output.permute(1, 0, 2))
        return output

class FormulaRecognitionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderTransformer(vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions[:, :-1])
        return outputs