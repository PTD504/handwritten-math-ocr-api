import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torchvision
from config import config

class EncoderSwin(nn.Module):
    def __init__(self):
        super().__init__()
        # Tải mô hình Swin Transformer Tiny với pretrained weights
        self.swin = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
        
        # Điều chỉnh đầu vào cho ảnh grayscale (1 kênh)
        original_conv = self.swin.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Khởi tạo weights mới bằng trung bình các kênh màu
        with torch.no_grad():
            new_conv.weight.copy_(torch.mean(original_conv.weight, dim=1, keepdim=True))
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)
        
        # Thay thế lớp conv đầu vào
        self.swin.features[0][0] = new_conv
        
        # Loại bỏ classification head
        self.features = self.swin.features
        
        # Projection layer để điều chỉnh chiều output
        self.projection = nn.Linear(768, config.d_model)  # 768 là output dim của Swin-T

    def forward(self, x):
        # Forward qua Swin Transformer
        x = self.features(x)  # [batch, H/32, W/32, 768]

        # Chuyển đổi kích thước: [batch, channels, height, width] -> [batch, (height*width), channels]
        batch, height, width, channels = x.shape        
        x = x.view(batch, height * width, channels)
        
        # Projection về chiều d_model
        x = self.projection(x)  # [batch, seq_len, d_model]
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding layer cho token đầu vào
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer Decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=config.swin_num_decoder_layers)
        
        # Output layer
        self.fc_out = nn.Linear(config.d_model, vocab_size)
        
        # Tạo mask cho sequence
        self.register_buffer("tgt_mask", self.generate_square_subsequent_mask(config.max_seq_len))

    def generate_square_subsequent_mask(self, sz):
        """Tạo mask tam giác trên để ngăn decoder nhìn thấy token tương lai"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, encoder_out, tgt):
        # Embedding + positional encoding
        tgt_embed = self.embedding(tgt)
        positions = torch.arange(0, tgt.size(1)).unsqueeze(0).to(tgt.device)
        tgt_embed = tgt_embed + self.pos_encoder(positions)
        
        # Chuẩn bị input cho decoder
        tgt_embed = tgt_embed.permute(1, 0, 2)  # [seq_len, batch, d_model]
        encoder_out = encoder_out.permute(1, 0, 2)  # [seq_len, batch, d_model]
        
        # Transformer decoder
        output = self.decoder(
            tgt_embed,
            encoder_out,
            tgt_mask=self.tgt_mask[:tgt.size(1), :tgt.size(1)]
        )
        
        # Output layer
        output = self.fc_out(output.permute(1, 0, 2))  # [batch, seq_len, vocab_size]
        return output


class FormulaRecognitionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EncoderSwin()
        self.decoder = DecoderTransformer(vocab_size)

    def forward(self, images, captions):
        # Encoder: xử lý hình ảnh
        features = self.encoder(images)  # [batch, seq_len, d_model]
        
        # Decoder: sinh chuỗi token
        outputs = self.decoder(features, captions[:, :-1])  # Bỏ token cuối cùng
        return outputs