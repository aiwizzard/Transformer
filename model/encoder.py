import torch.nn as nn

from model.embedding import Embeddings, PositonalEncoding
from model.attention import MultiHeadAttention
from model.ffnet import FFNet


class EncoderLayer(nn.Module):
    r"""EncoderLayer Class
    
    Args:
        head(int): Number of heads
        model_dim(int): size of the model
        ff_dim(int): Hidden layer size for the feed forward network
        dropout_rate(float): Probability for how much dropout to perform

    """
    def __init__(self, head=8, model_dim=512, ff_dim=2048, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, head, dropout_rate)
        self.ffnet = FFNet(model_dim, ff_dim, dropout_rate)

    def forward(self, source, source_mask):
        # self attention
        encoded = self.self_attention(source, source, source, source_mask)
        # feed forward network
        encoded = self.ffnet(encoded)
        return encoded


class Encoder(nn.Module):
    r"""Encoder class
    
    Args:
        vocab_size(int): Size of the vocabulary
        n_position(int): Upper bound for the positional encoding
        n_layers(int): Number of EncoderLayer layers
        head(int): Number of the heads
        model_dim(int): Size of the model
        ff_dim(int): Hidden layer size
        dropout_rate(float): Probability for how much dropout to perform

    Attributes:
        embedding: Embeddings class for input embedding
        positional_encoding: Positional encoding class
        layers: List containing `n_layers` of EncoderLayer class
        layer_norm: To perform layer normalization 
    """
    def __init__(
        self,
        vocab_size=32000,
        n_position=256,
        n_layers=6,
        head=8,
        model_dim=512,
        ff_dim=2048,
        dropout_rate=0.1,
    ):
        super(Encoder, self).__init__()
        self.embedding = Embeddings(vocab_size, model_dim)
        self.positional_encoding = PositonalEncoding(model_dim, n_position, dropout_rate)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(head, model_dim, ff_dim, dropout_rate)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, source, source_mask):
        # apply word embedding(x)
        out = self.embedding(source)
        # apply positional_encoding
        out = self.positional_encoding(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out = layer(out, source_mask)
        return out
