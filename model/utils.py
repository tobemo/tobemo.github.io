import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.optimizer import Optimizer


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


class ReducingCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    attenuation: float
    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0, attenuation: float = 1, last_epoch: int = -1, verbose: bool = False) -> None:
        self.attenuation = attenuation # super init calls step so set attenuation first
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
    
    def step(self, epoch: int | None = None) -> None:
        if ( self.attenuation != 1 ) and ( self.last_epoch > 0 ) and ( self.T_cur + 1 >= self.T_i ):
            self.base_lrs = [self.attenuation * lr for lr in self.base_lrs]
        super().step(epoch)

