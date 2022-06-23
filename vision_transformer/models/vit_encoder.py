from torch import nn
import torch.nn.functional as F

class ViTEncoder(nn.Module):
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

class ViTEncoderNormAfter(nn.Module):
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
        
        out, _ = self.attn(x, x, x)
        
        # First residual connection
        resid = x + out

        # Pass through MLP layer
        out = self.norm1(resid)
        out = F.gelu(self.fc1(out))
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)

        # Second residual connection
        out = out + resid
        out = self.norm2(x)
        
        return out