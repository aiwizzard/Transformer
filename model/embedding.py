import math
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    r""" Embeddings class

    Args:
        vocab_size(int): size of the vocabulary to create the embedding
        model_dim(int): dimension of the model to create the embedding

    Attributes:
        scale_factor(float): sqrt of model dim as used in the paper
    """
    def __init__(self, vocab_size, model_dim):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)            # Add pad_idx
        self.scale_factor = math.sqrt(model_dim)

    def forward(self, x:torch.Tensor):
        # First create the embedding and then scale
        x = self.embedding(x) * self.scale_factor
        return x
        

class PositonalEncoding(nn.Module):
    r""" Positional Encoding class

    Args:
        model_dim(int): dimension of the model
        n_position(int): maximum bound for the embedding space
        dropout_rate(float): Probability for how much dropout to perform
    """
    def __init__(self, model_dim, n_position=256, dropout_rate=0.1):
        super(PositonalEncoding, self).__init__()
        # Add a buffer to the module named position_table
        self.register_buffer('position_table', \
            self.get_sinusoid_encoding_table(model_dim, n_position))
        self.dropout = nn.Dropout(dropout_rate)

    def get_sinusoid_encoding_table(self, model_dim, n_position):
        r""" Create the sinusoid table
        Args:
            model_dim(int): dimension of the model
            n_position(int): maximum bound for the embedding space
        """
        position_table = torch.zeros(n_position, model_dim)
        position = torch.arange(0, n_position).float().unsqueeze(1)
        div_term = 10000 ** (torch.arange(0.0, model_dim, 2) / model_dim)
        # Calculate different embedding space according to the index
        position_table[:, 0::2] = torch.sin(position / div_term)
        position_table[:, 1::2] = torch.cos(position / div_term)
        position_table = position_table.unsqueeze(0)

        return position_table

    def forward(self, x: torch.Tensor):
        x = x + self.position_table[:, x.size(1), :]
        x = self.dropout(x)
        return x
