import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from vision_transformer.models.vit_encoder import ViTEncoder

def img_to_patch(x, patch_size):
    '''Transforms image into list of patches of the specified dimensions

    Args:
        x (Tensor): Tensor of dimensions B x C x H x W, representing a batch.
        B=Batch size, C=Channel count.
        patch_size (int): Size of one side of (square) patch.

    Returns:
        patch_seq (Tensor): List of patches of dimension B x N x [C * P ** 2],
        where N is the number of patches and P is patch_size.
    '''
    B, C, H, W = x.shape

    # reshape to B x C x H_count x H_patch x W_count x W_patch
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.flatten(1,2)
    x = x.flatten(2, 4)
    
    return x

class ViTClassifier(nn.Module):
    '''Encoder-only vision transformer

    Args:
        embed_dim (int): Size of embedding output from linear projection layer
        hidden_dim (int): Size of MLP head
        class_head_dim (int): Size of classification head
        num_encoders (int): Number of encoder layers
        num_heads (int): Number of self-attention heads
        patch_size (int): Size of patches
        num_patches (int): Total count of patches (patch sequence size) 
        dropout (float): Probability of dropout
    '''
    def __init__(
        self, embed_size, hidden_size, class_head_dim, num_encoders, 
        num_heads, patch_size, num_patches, dropout):
        super().__init__()

        # Key parameters
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Initial projection of flattened patches into an embedding
        self.input = nn.Linear(3*(patch_size**2), embed_size)
        self.drop = nn.Dropout(dropout)

        # Transformer with arbitrary number of encoders, heads, and hidden size
        self.transformer = nn.Sequential(
            *(ViTEncoder(embed_size, hidden_size, num_heads, dropout) for _ in range(num_encoders))
        )
        
        # Classification head
        self.fc1 = nn.Linear(embed_size, class_head_dim)
        self.fc2 = nn.Linear(class_head_dim, 100)

        # Learnable parameters for class and position embedding
        self.class_embed = nn.Parameter(torch.randn(1, 1, embed_size))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_size))

    def forward(self, x):
        # x will be in the shape B x N x C x P x P
        x = img_to_patch(x, self.patch_size)       

        # pass input through projection layer; shape is B x N x (C * P**2)
        x = F.relu(self.input(x))
        B, N, L = x.shape

        # concatenate class embedding and add positional encoding
        class_embed = self.class_embed.repeat(B, 1, 1)
        x = torch.cat([class_embed, x], dim=1)
        x = x + self.pos_embed[:, :N+1]
        x = self.drop(x)

        # apply transformer
        x = x.transpose(0, 1) # result is N x B x (C * P**2)
        x = self.transformer(x)
        x = x[0] # grab the class embedding
        
        # pass through classification head
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x