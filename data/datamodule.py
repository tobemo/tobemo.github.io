from typing import Any, Dict

import lightning as L
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from sklearn.preprocessing import Normalizer, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.dataset import RandomDataset


class BoringDataModule(L.LightningDataModule):
    train_set: Dataset
    valid_set: Dataset
    test_set: Dataset
    predict_set: Dataset
    scaler: StandardScaler | RobustScaler | Normalizer = None
    
    def __init__(self, size: int, length: int, batch_size: int = 32 ,**kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
    
    def prepare_data(self) -> None:
        # called once
        pass
    
    def setup(self, stage: str = '') -> None:
        # called once per machine
        if stage == 'fit':
            self.train_set = RandomDataset(size=self.hparams.size, length=self.hparams.length)
        if stage in ("fit", "validate"):
            self.valid_set = RandomDataset(size=self.hparams.size, length=self.hparams.length)
        if stage == "test":
            self.test_set = RandomDataset(size=self.hparams.size, length=self.hparams.length)
        if stage == "predict":
            self.predict_set = RandomDataset(size=self.hparams.size, length=self.hparams.length)
    
    def state_dict(self) -> Dict[str, Any]:
        # store in checkpoint
        state_dict = super().state_dict()
        state_dict['scaler'] = self.scaler
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # load from checkpoint
        self.scaler = state_dict['scaler']
    
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
            dataset=self.valid_set,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True,
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.hparams.batch_size,
        )
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.predict_dataloader,
            batch_size=self.hparams.batch_size,
        )

