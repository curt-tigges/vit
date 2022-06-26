import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import vision_transformer.models.vit_classifier as vc

from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

class ViTTrainModule(pl.LightningModule):
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
        batch_size (int): Batch size (used for OneCycleLR)
        learning_rate (float): Maximum learning rate
        weight_decay (float): Optimizer weight decay
    '''
    def __init__(
        self, 
        embed_dim, 
        hidden_dim, 
        class_head_dim, 
        num_encoders, 
        num_heads, 
        patch_size, 
        num_patches, 
        dropout, 
        batch_size, 
        learning_rate=0.001,
        weight_decay=0.03):
        super().__init__()

        # Key parameters
        self.save_hyperparameters()

        # Transformer with arbitrary number of encoders, heads, and hidden size
        self.model = vc.ViTClassifier(
            embed_dim,
            hidden_dim,
            class_head_dim,
            num_encoders,
            num_heads,
            patch_size,
            num_patches,
            dropout
        )

    def forward(self, x):
        x = self.model(x)        
        return x

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            betas=(0.9,0.999),
            weight_decay=self.hparams.weight_decay)
        
        steps_per_epoch = 60000 // self.hparams.batch_size
   
        lr_scheduler_dict = {
            "scheduler":OneCycleLR(
                optimizer,
                self.hparams.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
                anneal_strategy='cos'
            ),
            "interval":"step",
        }
        return {"optimizer":optimizer, "lr_scheduler":lr_scheduler_dict}