from typing import Any

import adabelief_pytorch
import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from model.rocket import Rocket
from model.utils import ReducingCosineAnnealingWarmRestarts


def get_layer_sizes(start: int, end: int, num: int) -> list[int]:
    """Returns exponentially decreasing layer sizes."""
    if end > start:
        raise ValueError(f"Start {start} should be greater than end {end}.")
    if num == 1:
        return [start, end]
    sizes: np.ndarray = np.logspace(np.log2(end), np.log2(start), num=num, base=2)
    sizes = sizes.astype(int)
    sizes[0] = end
    sizes[-1] = start
    sizes = sizes[::-1]
    return sizes.tolist()


class BaseClassifier(LightningModule):
    learner: Rocket
    classifier: torch.nn.Linear
    _lr_T_mult: int = 1
    """each cycle takes x as long for lr to decrease"""
    _lr_att: float = 0.4
    """multiplied with lr every time it restarts"""
    def __init__(self, learner: Rocket, C: int, n_layers: int = 1, lr: float = 1e-2, lr_restart: int = 4, dropout: float = 0.2) -> None:
        """A learner-classifier combination where a linear layer is fitted on the output of learner.

        Args:
            learner (Rocket): The feature extractor.
            C (int): The number of classes.
            n_layers (int, optional): How deep the linear layer is. Defaults to 1.
            lr (float, optional): The global learning rate. Defaults to 1e-2.
            lr_restart (int, optional): After how many cycles the learning rate increases.
            See CosineAnealingWithRestarts. Defaults to 4.
            dropout (float, optional): Dropout to use with the linear layers. Defaults to 0.2.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['learner'])
        
        self.learner = learner
        intermediate_shape = self.learner.shape[-1] * self.learner.shape[-2]
          
        layers = get_layer_sizes(intermediate_shape, C, n_layers)
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(layers[0])
        )
        for in_size, out_size in zip(layers[0:], layers[1:]):
            self.classifier.append(torch.nn.Dropout(dropout))
            self.classifier.append(torch.nn.Linear(in_size, out_size, bias=True))
            if out_size == C:
                break
            self.classifier.append(torch.nn.SiLU())
    
    def forward(self, X: torch.Tensor) -> Any:
        X_ = self.learner(X)
        X_ = torch.flatten(X_, start_dim=-2, end_dim=-1)
        y_ = self.classifier(X_)
        y_ = F.log_softmax(y_, dim=-1)
        return y_

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams, {"hp/loss": 0, "hp/acc": 0})
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = adabelief_pytorch.AdaBelief(
            self.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = ReducingCosineAnnealingWarmRestarts(
            attenuation=self._lr_att,
            optimizer=optimizer,
            T_0=self.hparams.lr_restart,
            T_mult=self._lr_T_mult,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": 'epoch',
                "frequency": 1,
            },
        }
    
    def generic_step(self, X: torch.Tensor, y_true: torch.Tensor) -> STEP_OUTPUT:
        logits = self.forward(X)
        loss = F.nll_loss(logits, y_true)
        y_pred = torch.argmax(logits, dim=1)
        return loss, y_pred
    
    def training_step(self, batch: torch.Tensor, batch_idx: int = None) -> STEP_OUTPUT:
        X, y_true = batch
        loss, _ = self.generic_step(X, y_true)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int = None) -> STEP_OUTPUT:
        X, y_true = batch
        loss, y_pred = self.generic_step(X, y_true)
        acc = accuracy(preds=y_pred, target=y_true, task='multiclass', num_classes=self.hparams.C)
        self.log("hp/loss", loss)
        self.log("hp/acc", acc)
        self.log("valid/loss", loss)
        self.log("valid/acc", acc)
        return loss
    
    def test_step(self, batch: torch.Tensor, batch_idx: int = None) -> STEP_OUTPUT:
        X, y_true = batch
        loss, y_pred = self.generic_step(X, y_true)
        acc = accuracy(preds=y_pred, target=y_true, task='multiclass', num_classes=self.hparams.C)
        self.log("test/loss", loss)
        self.log("test/acc", acc)
        return loss
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int = None, dataloader_idx: int = 0) -> STEP_OUTPUT:
        X, y_true = batch
        _, y_pred = self.generic_step(X, y_true)
        return y_pred

