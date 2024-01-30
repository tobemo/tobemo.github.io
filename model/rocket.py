"""
Inspired by:
    - Rocket, MiniRocket, MultiRocket and QUANT, from Dempster et al.

adapted from:
    - https://github.com/ChangWeiTan/MultiRocket/blob/main/multirocket/multirocket_multivariate.py
"""

import warnings
from itertools import combinations

import numpy as np
import torch
from torch import nn


def get_dilations(n_kernels: int, input_length: int, hidden_dimension: int, max_dilations_per_kernel: int, kernel_length: int = 9) -> tuple[list, np.ndarray]:
    """Get an exponentially increasing set of dilations.
     It still fits with the input length and won't be greater than max_dilations_per_kernel.
    Secondly a suggested number of bias terms for each dilation is returned.
     This divides hidden_dimension over each dilation, with a preference towards smaller dilations.
    
    Max dilations can lower the actual output size.

    Args:
        n_kernels (int): Number of kernels that will be used.
        input_length (int): Length of input time series.
        hidden_dimension (int): How many kernels/filters (dilation-bias combinations) are approximately generated (<=).
        max_dilations_per_kernel (int): The largest allowed dilation. Too large of a dilation might be overkill.

    Raises:
        ValueError: hidden_dimension should be greater than n_kernels.

    Returns:
        tuple[list, np.ndarray]: A tuple with a list of the suggested dilations
         and a list with the amount of kernels needed for each dilation to get the desired output size.
    """
    if hidden_dimension < n_kernels:
        raise ValueError(f"Hidden dimension should be greater than {n_kernels}, is {hidden_dimension}.\
                         Increase hidden dimension or decrease number of input channels.")
    
    if kernel_length != 9:
        raise NotImplementedError("Only a kernel length of 9 is supported for now.")
    if input_length <= kernel_length:
        raise ValueError(f"input_length ({input_length}) should be greater than kernel_length ({kernel_length}).")

    num_features_per_kernel = hidden_dimension // n_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (kernel_length - 1))
    dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
                  return_counts=True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


def generate_bias(k: int) -> np.ndarray:
    return np.linspace(start=-1, stop=1, num=k)

def generate_bias_set(n: int, k: int) -> np.ndarray:
    """Generate consistent biases.

    Args:
        n (int): Number of kernels.
        k (int): Number of bias sets. Each set has the same bias term 
        repeated n times.

    Returns:
        np.ndarray: Biases of shape (k,n) with repeated values along second axis.
    """
    biases = generate_bias(k)
    bias_set = np.tile(biases, n).reshape(n, k).T
    return bias_set


def get_bias(n_kernels: int, num_features_per_dilation: np.ndarray) -> list[np.ndarray]:
    """Uniformly sample bias terms from [-1, 1).
    Bias terms are returned in a format compatible with `get_dilations`.
    Namely a list of the same length as the amount of dilations is returned.
    Each element is a 

    Args:
        n_kernels (int): Number of kernels that will be used.
        num_features_per_dilation (np.ndarray): A list defining how many bias terms each dilation gets.
         See second return value of `get_dilations`.

    Returns:
        list[np.ndarray]: A list of 2D arrays.
         The second dimension of the bias terms should correspond to the first dimension of the weights matrix.
         The first dimension corresponds to the amount of bias terms generated for a dilation.
         So if a certain dilation only has 1 corresponding bias term it will be a 1xN array.
         If it has 5 corresponding bias terms it will be a 5xN array.
    """
    biases = []
    for n in num_features_per_dilation:
        bias = generate_bias_set(n=n_kernels, k=n)
        bias = bias.astype(np.float32)
        biases.append(bias)
    return biases


def get_kernel_parameters(input_length: int, n_kernels: int = 84, hidden_dimension: int = 10_000, max_dilations_per_kernel=32) -> tuple[list[int], list[np.ndarray]]:
    """Get all possible combinations that can be used for each kernel.
    This includes an exponentially increasing collection of dilations 
     plus a set of biases for each dilation.
    Smaller dilations will have more bias terms.
    
    The idea is to get a bunch of dilation-bias combinations
     to use for each kernel.

    Args:
        input_length (int): Input length along the time dimension.
        n_kernels (int, optional): Number of kernels that will be used. Defaults to 84.
        hidden_dimension (int, optional): How many kernels/filters (dilation-bias combinations) are approximately generated (<=).
         This influences the number if bias terms generated. Defaults to 10_000.
        max_dilations_per_kernel (int, optional): Max dilation allowed. This can lower the number of generated features. Defaults to 32.

    Returns:
        tuple[list[int], list[np.ndarray]]: A tuple containing a list with all the suggested dilation values
         and 2D array containing all the bias terms for said dilations.
         See `get_dilations` and `get_bias` respectively.
    """
    dilations, num_features_per_dilation = get_dilations(
        n_kernels=n_kernels,
        input_length=input_length,
        hidden_dimension=hidden_dimension,
        max_dilations_per_kernel=max_dilations_per_kernel
    )

    biases = get_bias(n_kernels, num_features_per_dilation)

    return dilations, biases


def get_kernel_weights(n_kernels: int = 84, kernel_length: int = 9) -> np.ndarray:
    """Pre-compute all possible kernels.
    Default values are recommended.
    
    A kernel consists of -1s and 2s and should sum to 0.
    With 84 kernels for a length of 9 each possible combination is made."""
    
    if kernel_length != 9:
        raise NotImplementedError("Only a kernel length of 9 is supported for now.")
    
    beta_lenght = 3
    alpha_kernel = np.ones((n_kernels, kernel_length), dtype=np.float32) * -1
    beta_indices = np.array(
        [_ for _ in combinations(np.arange(kernel_length), beta_lenght)],
        dtype = np.int32
    )
    
    # https://stackoverflow.com/a/20104162
    alpha_kernel[np.arange(len(beta_indices))[:,None], beta_indices] = 2
    return alpha_kernel


class Rocket(nn.Module):
    """Use a fixed set of kernels (default 84) to find and describe patterns in a time-series input.
    A dynamic set of dilations is computed in relation to the input length.
    For each dilation a number of bias terms is computed to get the desired hidden dimension or slightly less.
    (Max dilations per kernel can influence this number.)
    
    Each kernel quantifies the extend of a pattern match.
    The input is an N-dimensional time series,
     the output size depends on the input length and can be retrieved via `self.shape`.
    Each input channel is analyzed independently."""
    kernel_shape: tuple[int, int] = (84, 9)
    @property
    def kernels(self) -> int:
        return self.kernel_shape[0]
    @property
    def kernel_size(self) -> int:
        return self.kernel_shape[-1]
    
    in_channels: int
    conv: nn.Sequential | list[nn.Conv1d]
    
    @property
    def shape(self) -> tuple[int]:
        return (
            -1,
            self.conv[-1].weight.shape[0] * self.in_channels,
            -1,
        )
    
    def __init__(self, max_in_length: int, in_channels: int, hidden_dimension: int = None, max_dilations_per_kernel: int = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        hidden_dimension = hidden_dimension or 10_000
        max_dilations_per_kernel = max_dilations_per_kernel or 64
        
        # get a set of dilations that fit within 'max_in_length' and a number of biases that approaches 'hidden_dimension / len(dilations)'
        dilations, biases = get_kernel_parameters(
            input_length=max_in_length,
            n_kernels=self.kernels,
            hidden_dimension=hidden_dimension // in_channels,
            max_dilations_per_kernel=max_dilations_per_kernel,
        )
        
        # there are multiple biases per dilation (bias sets)
        # the goal is to stack said biases so that, using grouping, 
        # only one convolution per dilation is needed
        # compared to doing a convolution for each bias for each dilation
        biases = [torch.from_numpy(bias) for bias in biases]
        
        # although weights is one constant 2D array it will need to be repeated
        # so that both each channel and each bias set gets multiplied by a weight term
        weights = get_kernel_weights(self.kernels)
        weights = torch.from_numpy(weights)
        weights = weights.unsqueeze(1)
        
        # the amount of padding needed to account for dilation is 4 * dilation
        # this is derived from the formula of the output shape of a 1D conv
        paddings = [4 * d for d in dilations]
        max_pad = max(paddings)

        # dilation 1 has less padding needed than say dilation 3
        # weights should be centered in the massive kernel to 
        # create even padding on both sides
        # this can be done by shifting by the excess padding for each dilation
        padding_excess = [max_pad - p for p in paddings]
        
        # a massive kernel combines multiple kernels (with differing dilations) into one
        # this is only possible because a constant, non-learning, kernel is used
        # dilations can then be emulated by using 0s in between weights; e.g. w, 0, w is a dilation of 2
        out_channels = self.kernels * sum([b.shape[0] for b in biases])
        large_kernel = torch.zeros(out_channels, 1, (self.kernel_size * max(dilations)))
        large_bias = torch.cat(biases).flatten()
        
        # for each dilation
        i = 0
        for dilation, biases_for_this_dilation, excess in zip(dilations, biases, padding_excess):
            # for each bias for this dilation
            n_versions = biases_for_this_dilation.shape[0]
            for j in range(n_versions):
                slice_ = np.s_[
                    (i+j) * self.kernels : (i+j+1) * self.kernels,              # every 84 rows
                    :,                                                          # all channels
                    excess : excess + self.kernel_size * dilation : dilation,   # 9 indices, centered using excess ; in steps of 'dilation'
                ]
                large_kernel[slice_] = weights
            
            i += j + 1
        
        # write into 1 conv module
        massive_conv = torch.nn.Conv1d(
            in_channels=1, # see self._conv()
            out_channels=out_channels,
            kernel_size=self.kernel_size * max(dilations),
            stride=1,
            dilation=1,
            bias=True,
        )
        massive_conv.requires_grad_(False)
        massive_conv.weight.copy_(large_kernel)
        massive_conv.bias.copy_(large_bias)
        self.conv = nn.Sequential(
            nn.ZeroPad1d(max_pad),
            massive_conv
        )
    
    def _conv(self, X: torch.Tensor) -> torch.Tensor:
        """Treat each channel separately"""
        X_ = [self.conv(X[:,[i],:]) for i in range(X.shape[1])]
        return torch.cat(X_, dim=1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._conv(X)


class TemporalRocket(Rocket):
    """Alias for Rocket.
    
    Rocket is performed along the time dimension."""
    def __init__(self, max_in_length: int, in_channels: int = 1, hidden_dimension: int = None, max_dilations_per_kernel: int = None) -> None:
        super().__init__(max_in_length, in_channels, hidden_dimension, max_dilations_per_kernel)


class SpatialRocket(Rocket):
    """Spatial Rocket Module.
    
    Rocket is performed along feature/channel dimension."""
    @property
    def shape(self) -> tuple[int]:
        shape = super().shape
        return tuple(reversed(shape))
    
    def __init__(self, max_in_length: int, in_channels: int = 1, hidden_dimension: int = None, max_dilations_per_kernel: int = None) -> None:
        try:
            super().__init__(in_channels, max_in_length, hidden_dimension, max_dilations_per_kernel)
        except ValueError as e:
            if "Hidden dimension is too small" in str(e):
                raise ValueError("Hidden dimension is too small. Increase hidden dimension or decrease length of input.")
            elif "input_length" in str(e):
                raise ValueError(f"in_channels ({in_channels}) should be greater than kernel_length (default 9).")
            else:
                raise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        warnings.warn("Not tested.")
        y = super().forward(x.mT)
        return y.mT

