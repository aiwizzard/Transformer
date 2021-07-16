import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """ MultiHead attention.
    
    Perform Scaled dot product attention with different learned linear projection
    of queries, keys and values of dimension k_dim(model_dim / head) for head times
    in parallel.

    Args:
        model_dim(int): output dimension
        head(int): Number of heads. This is for performing attention function in parallel
        dropout_rate(float): Probability for how much dropout to perform

    Attributes:
        qkv_weights(Tensor): Learnable weights for Linear transformation for query, key and value since
        they have the same dimension(model_dim).
        dense(Tensor): Dense Layer
        attention(Tensor): Scaled dot product attention
        dropout : nn.Dropout class to regularize output
        layer_norm: To apply layer normalization. This is for keeping the weights
        with mean 0 and variance 1, so that the weights are normalized.    
    """
    def __init__(self, model_dim=512, head=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        # To ensure that parallelization works
        assert model_dim % head == 0, "invalid model dimension or number of heads"
        self.head = head
        self.key_dim = model_dim // self.head

        # To apply the linear transformation to the incomming data
        self.qkv_weights = nn.Linear(model_dim, model_dim, False)

        self.dense = nn.Linear(model_dim, model_dim, False)
        # Scaled dot product attention
        self.attention = ScaledDotProductAttention(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        residual = query
        # Project query, key and value to perform parallel attention in different heads
        query = (
            self.qkv_weights(query)
            .contiguous()
            .view(batch_size, -1, self.head, self.key_dim)
            .transpose(1, 2)
        )
        key = (
            self.qkv_weights(key)
            .contiguous()
            .view(batch_size, -1, self.head, self.key_dim)
            .transpose(1, 2)
        )
        value = (
            self.qkv_weights(value)
            .contiguous()
            .view(batch_size, -1, self.head, self.key_dim)
            .transpose(1, 2)
        )

        out = self.attention(query, key, value, mask)
        # Transpose to move the head dimension back and
        # combine the last two dimensions to concatenate all the heads together
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.head * self.key_dim)
        )

        out = self.dropout(self.dense(out))
        out += residual
        out = self.layer_norm(out)

        return out
        


class ScaledDotProductAttention(nn.Module):
    r"""Scaled dot product attention. 

    Compute the dot products of the query with all the keys, devided each by
    sqrt(k_dim), and apply a softmax function to obtain the weights on the values.

    Args:
        dropout_rate(float): Probability for how much dropout to perform

    Attributes:
        dropout : nn.Dropout class to regularize attention
    """
    def __init__(self, dropout_rate=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        # Get the dimension of query
        k_dim = query.size(-1)
        # Calculate matrix multiplication with query and key and then scale the
        # result by sqrt of dimension of query(key)
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_dim)
        # Apply mask
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        # Calculate softmax probability distribution
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        if self.dropout_rate is not None:
            attention = self.dropout(attention)
        # Calculate matrix multiplication of attention with value
        output = torch.matmul(attention, value)

        return output
