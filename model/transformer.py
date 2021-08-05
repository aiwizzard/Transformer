import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.generator import Generator


class Transformer(nn.Module):
    r"""Transformer Class
    
    Args:
        config: Contains model configuration

    Attributes:
        encoder: Encoder layer
        decoder: Decoder layer
        generator: Generate probability distribution
    """
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            config['vocab_size'],
            config['n_position'],
            config['n_layers'],
            config['head'],
            config['model_dim'],
            config['ff_dim'],
            config['dropout_rate'],
        )
        self.decoder = Decoder(
            config['vocab_size'],
            config['n_layers'],
            config['head'],
            config['model_dim'],
            config['n_position'],
            config['ff_dim'],
            config['dropout_rate'],
        )
        self.generator = Generator(config['model_dim'], config['vocab_size'])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, source, source_mask):
        encoded = self.encoder(source, source_mask)
        return encoded

    def decode(self, target, encoded, source_mask, target_mask):
        decoded = self.decoder(target, encoded, source_mask, target_mask)
        return decoded

    def generate(self, x):
        return self.generator(x)

    def forward(self, source, source_mask, target, target_mask):
        encoded = self.encode(source, source_mask)
        decoded = self.decode(target, encoded, source_mask, target_mask)
        out = self.generate(decoded)
        return out
