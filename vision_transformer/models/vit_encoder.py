from torch import nn
import torch.nn.functional as F

class ViTEncoder(nn.Module):
    '''Basic transformer encoder, as specified in the paper

    Args:
        input_dim (int): Dimensions of transformer input (input embed size)
        hidden_dim (int): Size of MLP head
        num_heads (int): Number of self-attention heads
        dropout (float): Probability of dropout
    '''
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads)
        self.norm2 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.norm1(x)
        out, _ = self.attn(out, out, out)
        
        # First residual connection
        resid = x + out

        # Pass through MLP layer
        out = self.norm2(resid)
        out = F.gelu(self.fc1(out))
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)

        # Second residual connection
        out = out + resid

        return out