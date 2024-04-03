import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequentialNumberBase(Dataset):
    data: pd.DataFrame
    data_raw: pd.DataFrame
    """data without scaling"""
    sequence_length: int
    mean: float = None
    std: float = None
    
    def __init__(self, train: bool = True, length: int = None, pad: int = 0, mean: float = 0, std: float = 1) -> None:
        assert hasattr(self, "data"), "Attribute 'data' should have been set in inherited class."
        assert isinstance(self.data, pd.DataFrame)
        self.data_raw = self.data.copy()
        
        # ensure time dimension is not longer than self.length
        self.sequence_length = self.data.shape[1] + pad if length is None else length + pad
        self.data = self.data.iloc[:, :self.sequence_length]
        
        # scale
        if train:
            self.compute_scaling()
        else:
            self.mean = mean
            self.std = std
        self.apply_scaling()
        
        # zero pad to a requested length
        necessary_padding = self.sequence_length - self.data.shape[-1]        
        if necessary_padding > 0:
            self.data = self.nan_pad(necessary_padding)
        
        self.data = self.data.fillna(0.)
        self.data = self.data.astype(np.float32)
    
    def nan_pad(self, padding: int) -> pd.DataFrame:
        """Zero pad tail (column wise)."""
        padding = pd.DataFrame(
            data=np.nan,
            index=self.data.index,
            columns=pd.RangeIndex(self.data.columns[-1] + 1, self.data.columns[-1] + padding + 1),
            dtype=np.float32,
        ) 
        data = pd.concat([self.data, padding], axis=1)
        return data
    
    def compute_scaling(self) -> None:
        """Compute global mean and standard deviation."""
        self.mean = self.data.stack(future_stack=True).dropna().mean()
        self.std = self.data.stack(future_stack=True).dropna().std()

    def get_scaling(self) -> tuple[float, float]:
        if self.mean is None or self.std is None:
            self.compute_scaling()
        return ( self.mean, self.std )
    
    def apply_scaling(self, mean: float = None, std: float = None) -> None:
        """Scale data.
        Args:
            mean (float, optional): Scale with a mean different than the one set by compute_scaling. Defaults to None.
            std (float, optional): Scale with a standard deviation different than the one set by compute_scaling. Defaults to None.

        Raises:
            ValueError: When neither passed arguments or attribute _mean and _std differ from None.
        """
        mean = self.mean if mean is None else mean
        std = self.std if std is None else std
        if mean is None or std is None:
            raise ValueError("One of mean or std is None. Has compute_scaling been called?")
        
        std = 1 if std == 0. else std
        self.data = (self.data - self.mean) / self.std
    
    def __len__(self) -> int:
        return max(
            self.data.index.get_level_values('id')
        ) + 1
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.data.loc[idx]
        target = sample.index[0][0]
        return torch.from_numpy(sample.to_numpy()), torch.tensor(target)


class SequentialNumberDataset(SequentialNumberBase):
    def __init__(self, train: bool = True, length: int = None, pad: int = 0, mean: float = 0, std: float = 1) -> None:
        self.data = pd.read_parquet(f"data/{'train' if train else 'test'}.parquet")
        super().__init__(train, length, pad, mean, std)


def dropout(shape: tuple[int, int], p: float = 0.1) -> None:
    rng = np.random.default_rng()
    other = np.ones(shape=shape)
    
    # reduce 
    to_zero = rng.choice(np.arange(other.size), replace=False, size=int(p * other.size))
    other[np.unravel_index(to_zero, other.shape)] = 0
    return other


def _new_ids(df: pd.DataFrame) -> pd.DataFrame:
    df.index = df.index.droplevel('id') # need new id else duplicates
    new_ids = pd.Index( np.arange(len(df)) // 2, name='id' )
    df = df.set_index(new_ids, append=True)
    df = df.reorder_levels(['id', 'number', 'direction'], axis=0)
    return df


class AugmentedSequentialNumberDataset(SequentialNumberBase):
    def __init__(self, train: bool = True, noise: float = 0.1, length: int = None, pad: int = 0, mean: float = 0, std: float = 1) -> None:
        # base dataset
        data = pd.read_parquet(f"data/{'train' if train else 'test'}.parquet")
        
        # noisy dataset
        noise = dropout(data.shape, p=noise)
        noisy_data = data * noise
        
        # reversed data set (strokes go in the opposite direction)
        reverse_data = pd.read_parquet(f"data/{'train' if train else 'test'}_reversed.parquet")
        
        # combine
        self.data = pd.concat(
            [
                data, noisy_data, reverse_data
            ], axis=0
        )
        
        # reset ids
        self.data = _new_ids(self.data)
        
        self.original_data = data
        self.noisy_data = noisy_data
        self.reversed_data = reverse_data
        
        super().__init__(train, length, pad, mean, std)


class SlicedSequentialNumberDataset(SequentialNumberBase):
    def __init__(self, train: bool = True, n_cuts: int = 3, length: int = None, pad: int = 0, mean: float = 0, std: float = 1) -> None:
        #  base dataset
        data = pd.read_parquet(f"data/{'train' if train else 'test'}.parquet")
        data = data.dropna(axis=1, thresh=data.shape[0]//2)
        self.n_cuts = n_cuts
        
        # cut size
        slice_size = data.shape[1] // self.n_cuts
        cuts = list(range(slice_size, data.shape[1] ,slice_size))
        
        # cut
        dfs = []
        for c in cuts:
            x = data.copy()
            x.iloc[ : , c: ] = np.nan
            dfs.append(x)
        dfs = pd.concat(dfs, axis=0)
        
        # reset ids
        data = _new_ids(dfs)
        
        self.data = _new_ids(data)
        super().__init__(train, length, pad, mean, std)

