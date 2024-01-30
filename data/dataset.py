import torch
from torch import Tensor
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index], 0

    def __len__(self) -> int:
        return self.len

