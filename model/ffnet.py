import torch.nn as nn

class FFNet(nn.Module):
    r"""FFNet class

    This is a feed forward network. Uses convolutional layers as per the paper
    
    Args:
        model_dim(int): dimension of the model
        ff_dim(int): Hidden layer size
        dropout_rate(float): Probability for how much dropout to perform

    Attributes:
        relu: Activation function
    """
    def __init__(self, model_dim=512, ff_dim=2048, dropout_rate=0.1):
        super(FFNet, self).__init__()
        self.layer1 = nn.Conv1d(model_dim, ff_dim, 1)
        self.layer2 = nn.Conv1d(ff_dim, model_dim, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.layer1(x.transpose(1, 2))
        out = self.relu(out)
        out = self.layer2(out)
        out = self.dropout(out.transpose(1, 2))
        # Residual Connection
        out += x
        out = self.layer_norm(out)

        return out
