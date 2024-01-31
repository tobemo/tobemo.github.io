from typing import Any, Dict

import lightning as L
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.dataset import (AugmentedSequentialNumberDataset,
                          SequentialNumberDataset,
                          SlicedSequentialNumberDataset)


class SequentialNumberBase(L.LightningDataModule):
    train_set: Dataset
    test_set: Dataset
    scaler: StandardScaler | Normalizer
    mean: float = None
    std: float = None
    length: int = None
    
    def __init__(self, batch_size: int = 32, **kwargs) -> None:
        """x&y coordinate-changes of a pen stroke over time.

        Args:
            batch_size (int, optional): Defaults to 32.
        """
        super().__init__()
        self.save_hyperparameters()
        self.scaler = StandardScaler()
    
    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict['mean'] = self.mean
        state_dict['std'] = self.std
        state_dict['length'] = self.length
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.length = state_dict['length']
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True,
        )


class SequentialNumberDataModule(SequentialNumberBase):
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__(batch_size)
    
    def setup(self, stage: str = '') -> None:
        self.train_set = SequentialNumberDataset(train=True, pad=20)
        
        self.mean = self.train_set.mean if self.mean is None else self.mean
        self.std = self.train_set.std if self.std is None else self.std
        self.length = self.train_set.sequence_length if self.length is None else self.length
        
        self.test_set = SequentialNumberDataset(
            train=False,
            length=self.length,
            mean=self.mean,
            std=self.std
        )


class AugmentedSequentialNumberDataModule(SequentialNumberBase):
    def __init__(self, batch_size: int = 32, noise: float = 0.1) -> None:
        super().__init__(batch_size=batch_size, noise=noise)
    
    def setup(self, stage: str = '') -> None:
        self.train_set = AugmentedSequentialNumberDataset(
            train=True,
            pad=20,
            noise=self.hparams.noise
        )
        
        self.mean = self.train_set.mean if self.mean is None else self.mean
        self.std = self.train_set.std if self.std is None else self.std
        self.length = self.train_set.sequence_length if self.length is None else self.length
        
        self.test_set = SequentialNumberDataset(
            train=False,
            length=self.length,
            mean=self.mean,
            std=self.std
        )


class SlicedSequentialNumberDataModule(SequentialNumberBase):
    def __init__(self, batch_size: int = 32, n_cuts: int = 3) -> None:
        super().__init__(batch_size=batch_size, n_cuts=n_cuts)
    
    def setup(self, stage: str = '') -> None:
        self.train_set = SlicedSequentialNumberDataset(
            train=True,
            pad=20,
            n_cuts=self.hparams.n_cuts
        )
        
        self.mean = self.train_set.mean if self.mean is None else self.mean
        self.std = self.train_set.std if self.std is None else self.std
        self.length = self.train_set.sequence_length if self.length is None else self.length
        
        self.test_set = SequentialNumberDataset(
            train=False,
            length=self.length,
            mean=self.mean,
            std=self.std
        )

