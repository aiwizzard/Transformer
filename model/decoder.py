import torch.nn as nn

from model.ffnet import FFNet
from model.attention import MultiHeadAttention
from model.embedding import Embeddings, PositonalEncoding


class Decoder(nn.Module):
    r"""Encoder class
    
    Args:
        vocab_size(int): Size of the vocabulary
        n_position(int): Upper bound for the positional encoding
        n_layers(int): Number of `DecoderLayer` layers
        head(int): Number of the heads
        model_dim(int): Size of the model
        ff_dim(int): Hidden layer size
        dropout_rate(float): Probability for how much dropout to perform

    Attributes:
        embedding: Embeddings class for input embedding
        positional_encoding: Positional encoding class
        layers: List containing `n_layers` of DecoderLayer class
        layer_norm: To perform layer normalization 
    """
    def __init__(
        self,
        vocab_size=32000,
        n_layers=6,
        head=8,
        model_dim=512,
        n_position=256,
        ff_dim=2048,
        dropout_rate=0.1,
    ):
        super(Decoder, self).__init__()
        # decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(head, model_dim, ff_dim, dropout_rate)
                for _ in range(n_layers)
            ]
        )
        self.embedding = Embeddings(vocab_size, model_dim)
        self.postional_encoding = PositonalEncoding(model_dim, n_position, dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, memory, source_mask, target_mask):
                                                        # source_mask = source_mask.unsqueeze(-2)
        # apply word embedding
        out = self.embedding(x)
        # apply positional encoding
        out = self.postional_encoding(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out = layer(out, memory, source_mask, target_mask)
        return out


class DecoderLayer(nn.Module):
    r"""Decoder Layer Class
    
    Args:
        head(int): Number of heads
        model_dim(int): size of the model
        ff_dim(int): Hidden layer size for the feed forward network
        dropout_rate(float): Probability for how much dropout to perform

    """
    def __init__(self, head=8, model_dim=512, ff_dim=2048, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        # Self Attention layer, query, key and value come from previous layer
        self.self_attention = MultiHeadAttention(model_dim, head, dropout_rate)
        # Source target attention layer, query come from encoded space.
        # key and value come from previous self attention layer
        self.st_attention = MultiHeadAttention(model_dim, head, dropout_rate)
        self.ffnet = FFNet(model_dim, ff_dim, dropout_rate)

    def forward(self, x, mem, source_mask, target_mask):
        # self attention
        out = self.self_attention(x, x, x, target_mask)
        # soure target attention
        out = self.st_attention(out, mem, mem, source_mask)
        # feed forward network
        out = self.ffnet(out)
        return out
