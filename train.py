import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

# Import custom modules
from data.cifar100 import CIFAR100DataModule
from vision_transformer.models.pl_model import ViTModel

CIFAR = "/media/curttigges/project-files/datasets/cifar-100/"

pl.seed_everything(42)

hyperparameter_defaults = {
    "embed_size":256, 
    "hidden_size":512,
    "hidden_class_size":512, 
    "num_encoders":36,
    "num_heads":8,
    "patch_size":4,
    "num_patches":64,
    "dropout":0.1,
    "batch_size":256,
    "learning_rate":0.001,
    "weight_decay":0.03
}

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

data_module = CIFAR100DataModule(
    batch_size=model_kwargs["batch_size"], 
    num_workers=12,
    data_dir=CIFAR)


model = ViTModel(**model_kwargs)

wandb_logger = WandbLogger(project="vit-cifar100")
wandb_logger.watch(model, log="all")

trainer = Trainer(
    max_epochs=180,
    accelerator='gpu', 
    devices=1,
    logger=wandb_logger, 
    callbacks=[TQDMProgressBar(refresh_rate=10)])

trainer.fit(model, datamodule=data_module)
#wandb.finish()