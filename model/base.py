import adabelief_pytorch
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from model.utils import ReducingCosineAnnealingWarmRestarts


class BoringModel(LightningModule):    
    _lr_T_mult: int = 1
    """each cycle takes x as long for lr to decrease"""
    _lr_att: float = 0.4
    """multiplied with lr every time it restarts"""
    
    def __init__(self, C: int, latent_dim: int = None, lr: float = 1e-3, lr_restart: int = 4, dropout: float = 0.2) -> None:
        """A demo model.

        Args:
            C (int): The number of classes.
            latent_dim (int, optional): Number of units in the linear layer.
            lr (float, optional): The global learning rate. Defaults to 1e-3.
            lr_restart (int, optional): After how many cycles the learning rate resets.
            It resets to self._lr_att * last_lr with last_lr starting at lr.
            See CosineAnealingWithRestarts. Defaults to 4.
            dropout (float, optional): Dropout to use with the linear layers. Defaults to 0.2.
        """
        super().__init__()
        self.save_hyperparameters()
        latent_dim = latent_dim or 32
        self.layer = torch.nn.Linear(latent_dim, C)

    def forward(self, x):
        return self.layer(x)

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

