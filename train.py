import pytorch_lightning as pl
#import wandb #import if tracking is desired
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

# Import custom modules
from data.cifar100 import CIFAR100DataModule
from vision_transformer.models.vit_classifier import ViTClassifier
from vision_transformer.models.vit_pl_train_module import ViTTrainModule

'''Simple script for training ViT

All hyperparameters listed below can be modified. Script can be run as-is with 
"python train.py" and hyperparameters are set by default to highest-performing
values for CIFAR-100.

Args:
    None

Returns:
    None

'''
# Set this to your local CIFAR-100 directory.
CIFAR = "/media/curttigges/project-files/datasets/cifar-100/"

pl.seed_everything(42)

hyperparameter_defaults = {
    "embed_dim":256, 
    "hidden_dim":512,
    "class_head_dim":512, 
    "num_encoders":24,
    "num_heads":8,
    "patch_size":4,
    "num_patches":64,
    "dropout":0.1,
    "learning_rate":0.001,
    "batch_size":256,
    "learning_rate":0.001,
    "weight_decay":0.03
}

# Enable the below if you have WandB and wish to run a sweep
'''
wandb.init(config=hyperparameter_defaults)

config = wandb.config

model_kwargs = {
    "embed_size":256, 
    "hidden_size":512,
    "hidden_class_size":512, 
    "num_encoders":config.num_encoders,
    "num_heads":8,
    "patch_size":config.patch_size,
    "num_patches":(32**2//(config.patch_size**2)),
    "dropout":config.dropout,
    "batch_size":config.batch_size,
    "learning_rate":config.learning_rate,
    "weight_decay":config.weight_decay
}
'''

# Disable dictionary below if you want to run a hyperparameter sweep with WandB
model_kwargs = hyperparameter_defaults

data_module = CIFAR100DataModule(
    batch_size=model_kwargs["batch_size"], 
    num_workers=12,
    data_dir=CIFAR)


model = ViTTrainModule(**model_kwargs)

# Enable these lines if you want to log with WandB
#wandb_logger = WandbLogger(project="vit-cifar100")
#wandb_logger.watch(model, log="all")

trainer = Trainer(
    max_epochs=180,
    accelerator='gpu', 
    devices=1,
    #logger=wandb_logger, 
    callbacks=[TQDMProgressBar(refresh_rate=10)])

trainer.fit(model, datamodule=data_module)