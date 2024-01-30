from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from model.pooling.positive_values import MIPV, MPOV, MPV, PPV, SPV


class SpatialConv(nn.Module):
    """Learn spatial kernels that analyze relation between input features per timestep.
    Kernels have width 1 (and depth equal to number of features).
    This results in a new 'time-series" with the same amount of timesteps but
    a different amount of channels. The idea being that this new representation
    holds more information on how each input channel relates to each other."""
    in_channels: int
    out_channels: int
    activation: Callable
    dense: bool
    
    @property
    def shape(self) -> tuple[int, int]:
        return (1, self.out_channels, -1)
    
    """Concat input to output, much like DenseNet."""
    def __init__(self, in_channels: int, out_channels: int, dense: bool = False, activation: Callable = F.silu) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True, # see note below
        )
        # An important element of rocket are the bias terms. Rocket works by analyzing positive vs negative values
        # after the temporal convolution, which is partially dependant on the fixed bias terms of rocket.
        # The bias terms of rocket shifts the pos-neg threshold (of each conv) up or down so
        # by allowing the bias term of the spatial block to be learned this shift can be determined more
        # intelligently. This bias term will shift all the pseudo-time-series up or down to,
        # together with the fixed bias terms of rocket, be better aligned around 0. This only adds 
        # a small overhead since there are (or should be) less spatial kernels than temporal kernels.
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation # TODO: optimize with optuna (tanh, hardtanh, tanhshrink)
        self.dense = dense
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = self.conv(X)
        y = self.activation(y)
        if self.dense:
            y = torch.concat(
                tensors=[X, y],
                dim=-2,
            )
        return y


class PositivePooler(nn.Module):
    """Does one or more of the positive pooling operations.
    What pooling operations are used depend on the given pool_level.
    Operations are:
        - PPV or proportion of positive values;
        - MPV or mean of positive values;
        - SPV or standard deviation of positive values;
        - MPOV or mean of positive outlier values (>2-std);
        - MIPV* or mean of indices of positive values;
    
    *Breaks graph and can thus not be used when gradients need to propagate
    back to before this operation.
    """
    pool_level: int
    def __repr__(self):
        return f"{self.__class__.__name__}({self.pool_level})"
    
    @property
    def shape(self) -> tuple[int]:
        return (-1, self.pool_level, -1)
    
    def __init__(self, pool_level: int = 1) -> None:
        super().__init__()
        if pool_level < 1:
            raise ValueError(f"Pool level should exceed 0 (is {pool_level}).")
        self.pool_level = pool_level
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pooling = [PPV(X)]
        if self.pool_level > 1:
            m = MPV(X) # possibly needed
            pooling.append(m)
        if self.pool_level > 2:
            s = SPV(X) # possibly needed
            pooling.append(s)
        if self.pool_level > 3:
            o = MPOV(X, m, s)
            pooling.append(o)
        if self.pool_level > 4:
            pooling.append(MIPV(X))

        return torch.stack(pooling, dim=-2)

