import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    r"""Generator class

    Args:
        model_dim(int): dimension of the model
        vocab(int): vocab size
    """
    def __init__(self, model_dim, vocab):
        super(Generator, self).__init__()
        self.dense = nn.Linear(model_dim, vocab)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.dense(x)
        # Calculate probability distribution
        x = F.log_softmax(x, dim=1)
        return x
