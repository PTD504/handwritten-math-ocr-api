import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torchvision
from config import config

"""
Model Architecture:
Encoder: Swin Transformer (Tiny)
Decoder: Transformer
"""

class EncoderSwin(nn.Module):
    def __init__(self):
        super().__init__()
        # Swin-Tiny Transformer encoder
        self.swin = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
        
        original_conv = self.swin.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        with torch.no_grad():
            new_conv.weight.copy_(torch.mean(original_conv.weight, dim=1, keepdim=True))
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)
        
        self.swin.features[0][0] = new_conv
        self.features = self.swin.features
        
        self.projection = nn.Linear(768, config.d_model)

    def forward(self, x):
        x = self.features(x)

        batch, height, width, channels = x.shape        
        x = x.view(batch, height * width, channels)
        
        x = self.projection(x)  
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        
        self.pos_encoder = nn.Embedding(config.max_seq_len, config.d_model)
        
        decoder_layer = TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        
        self.fc_out = nn.Linear(config.d_model, vocab_size)
        
        self.register_buffer("tgt_mask", self.generate_square_subsequent_mask(config.max_seq_len))

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, encoder_out, tgt):
        tgt_embed = self.embedding(tgt)
        positions = torch.arange(0, tgt.size(1)).unsqueeze(0).to(tgt.device)
        tgt_embed = tgt_embed + self.pos_encoder(positions)
        
        tgt_embed = tgt_embed.permute(1, 0, 2)
        encoder_out = encoder_out.permute(1, 0, 2)
        

        output = self.decoder(
            tgt_embed,
            encoder_out,
            tgt_mask=self.tgt_mask[:tgt.size(1), :tgt.size(1)]
        )
        
        output = self.fc_out(output.permute(1, 0, 2))
        return output


class FormulaRecognitionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EncoderSwin()
        self.decoder = DecoderTransformer(vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        
        outputs = self.decoder(features, captions)
        return outputs