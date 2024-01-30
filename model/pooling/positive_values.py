"""
Inspired by:
    - Rocket and MultiRocket from Dempster et al.
"""

import numpy as np
import torch
import torch.nn.functional as F


def get_longest_positive_sequence(x: torch.Tensor) -> torch.Tensor:
    """See `get_longest_true_sequence`."""
    return get_longest_true_sequence( x > 0 )


def _get_longest_true_sequence(x: np.ndarray) -> np.ndarray:
    """Get length of longest True sequence in last dimension of x.
    
    Args:
        x[np.ndarray]: A boolean ND matrix.
    """
    # pad as to include edges
    is_true_padded = np.pad(x, ((0, 0), (0, 0), (1, 1)), constant_values=False)
    
    # mark start (1) and end (-1) of sequence
    diff = np.diff(is_true_padded.astype(int), axis=-1)

    # get indices of start and end, for all dimensions
    start_indices = np.where(diff == 1)
    end_indices = np.where(diff == -1)
    
    # get lengths of all sequences, only compare in last dimension
    lengths = end_indices[-1] - start_indices[-1]
    
    # prepare a zero matrix
    longest_sequence_per_row = np.zeros(x.shape[:-1], dtype=int)
    
    # find, for all dimensions but the last, the max length;
    # by comparing values in 'longest_sequence_per_row', which are all 0s by design,
    # against values in lengths; do this on the indices defined by start indices
    # https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
    # https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    np.maximum.at(longest_sequence_per_row, start_indices[:-1], lengths)
    
    return longest_sequence_per_row


def get_longest_true_sequence(x: torch.Tensor) -> torch.Tensor:
    """Get length of longest True sequence in last dimension of x.
    
    Args:
        x[torch.Tensor]: A boolean ND matrix.
    """
    # pad as to include edges
    is_true_padded = F.pad(x, (1, 1), mode='constant', value=False)
    
    # mark start (1) and end (-1) of sequence
    diff = torch.diff(is_true_padded.int(), axis=-1)

    # get indices of start and end of sequences, for all dimensions
    start_indices = torch.where(diff == 1)
    end_indices = torch.where(diff == -1)
    
    # get lengths of all sequences, only compare in last/time dimension
    lengths = end_indices[-1] - start_indices[-1]
    
    # prepare a zero matrix; this handles the edge case where there is
    # no true-sequence i.e. length = 0
    longest_sequence_per_timeseries = torch.zeros(*x.shape[:2], device=x.device).int()
    
    # locate all lengths; this only indexes the batch and channel dimension,
    # not the last/time dimension
    # this indexes all coordinates for which there is a length > 0
    # there might very well be duplicate indices, this happens when there is 
    # more than one true sequence 
    indices_with_length = torch.stack(start_indices[:-1])
    
    # group all repeated indices, these are time series that have multiple
    # true sequences for the same batch-channel combo
    _, groupby_indices = indices_with_length.unique(dim=1, return_inverse=True, sorted=False) # TODO: check influence of sorted
    
    # only keep the max of each batch-channel combo
    longest_sequence_flat = torch.zeros(groupby_indices.max() + 1, device=lengths.device, dtype=lengths.dtype)
    longest_sequence_flat.scatter_reduce_(
        src=lengths,
        index=groupby_indices,
        dim=-1,
        reduce='amax',
    )
    
    # drop duplicate indices since we now only have one length per batch-channel combo
    out_indices = indices_with_length.unique(dim=1, sorted=False)
    
    # store in the right spot
    longest_sequence_per_timeseries[out_indices[0], out_indices[1]] = longest_sequence_flat.int()
    
    return longest_sequence_per_timeseries


def n_pos(x):
    """Differentiable approximation of a positive count."""
    # https://www.reddit.com/r/pytorch/comments/pwx7p6/ideas_on_how_to_create_a_differentiable_loss/
    return torch.sum(torch.sigmoid(x), dim=-1)

def PPV(x: torch.Tensor) -> torch.Tensor:
    """Proportion of positive values."""
    ppv = n_pos(x) / x.shape[-1]
    return ppv


def mean_where_true(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    n_meet_condition = torch.sum(cond, dim=-1)
    mv = torch.sum(cond * x, dim=-1) / n_meet_condition
    # mv = torch.nan_to_num(mv, 0)
    return mv

def MPV(x: torch.Tensor) -> torch.Tensor:
    """Mean of positive values"""
    mpv = F.relu(x).mean(dim=-1)
    return mpv


def SPV(x: torch.Tensor) -> torch.Tensor:
    """Standard deviation of positive values"""
    stdev = F.relu(x).std(dim=-1)
    return stdev


def _mpov(x: torch.Tensor, *, mpv: torch.Tensor, spv: torch.Tensor) -> torch.Tensor:
    threshold = mpv + 2 * spv
    mpov = MPV(x - threshold.unsqueeze(-1))
    return mpov
    
def MPOV(x: torch.Tensor, mpv: torch.Tensor = None, spv: torch.Tensor = None) -> torch.Tensor:
    """Mean of positive-outlier values.
    Mean of positive values greater than two standard deviations of the average positive value."""
    mpv = mpv if mpv is not None else MPV(x)
    spv = spv if spv is not None else SPV(x)
    return _mpov(x, mpv=mpv, spv=spv)


def quantiles(x: torch.Tensor) -> torch.Tensor:
    return torch.quantile(
        x,
        torch.tensor([0.25, 0.5, 0.75], device=x.device),
        dim=-1
    )


def QM(x: torch.tensor, *, t_lower: torch.Tensor = None, t_upper: torch.Tensor = None) -> torch.Tensor:
    t_lower = torch.full(x.shape[:-1], -torch.inf, device=x.device) if t_lower is None else t_lower
    t_upper = torch.full(x.shape[:-1], +torch.inf, device=x.device) if t_upper is None else t_upper
    
    t_lower = t_lower.unsqueeze(-1).expand(x.shape) # add time dimension
    t_upper = t_upper.unsqueeze(-1).expand(x.shape) # add time dimension
    
    cond = torch.gt(x, t_lower) & torch.lt(x, t_upper)
    qm = mean_where_true(x, cond)
    return qm


def MIPV(x: torch.Tensor) -> torch.Tensor:
    """Mean of indices of positive values.
    Instead of marking the MIPV for all negative values with a -1 it is marked with a zero.
    The indices of all positive values are increased by 1.
    This has the benefit of including the first index as well, were it to be positive, else it would just be 0.
    (The first index does still influence the mean by increasing the denominator by one.)"""
    is_pos = x > 0
    n_pos = torch.sum(is_pos, dim=-1)
    indices = torch.arange(1, x.shape[-1] + 1, device=x.device).expand(*x.shape)
    mipv = torch.sum( is_pos * indices, dim=-1) / n_pos
    mipv = torch.nan_to_num(mipv, 0)
    return mipv


def LSPV(x: torch.Tensor) -> torch.Tensor:
    """Longest stretch of positive values."""
    is_pos = x > 0
    lspv = get_longest_true_sequence(is_pos)
    lspv = lspv.type(x.dtype)
    return lspv

