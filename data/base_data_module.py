from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
import argparse

from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl

class BaseDataModule(pl.LightningDataModule):
    '''
    '''

    def __init__(self, batch_size=128, num_workers=0, classes=1) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = classes

    def config(self):
        '''Returns attributes of the dataset that will be used to instantiate models.
        '''
        return {"output_dims": self.classes}


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """