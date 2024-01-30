from torch import nn

from model.base import BaseClassifier
from model.blocks import PositivePooler, SpatialConv
from model.rocket import TemporalRocket


class TemporalClassifier(BaseClassifier):    
    def __init__(self, input_length: int, input_channels: int, C: int, n_layers: int, pool_level: int, hidden_dimension: int = None, max_dilations_per_kernel: int = None, **kwargs) -> None:
        """Analyze each input channel independently over time and
            learn how all channels combined relate to what class.

        Args:
            input_length (int): Largest expected input slice.
            This bounds the max dilution used. Longer inputs will still work.
            input_channels (int): How many channels/features.
            C (int): How many classes to learn.
            n_layers (int): How deep the classifier is.
            pool_level (int): How many pooling operations to combine. See model.blocks.PositivePooler.
            temporal_kwargs: Are passed to the TemporalRocket.
            kwargs: Are passed to the BaseClassifier.
        """
        learner = nn.Sequential(
            TemporalRocket(
                max_in_length=input_length,
                in_channels=input_channels,
                hidden_dimension=hidden_dimension,
                max_dilations_per_kernel=max_dilations_per_kernel,
            ),
            PositivePooler(pool_level=pool_level),
        )
        shape = list(learner[-1].shape)
        shape[-1] = learner[-2].shape[-2]
        learner.shape = shape
        
        super().__init__(learner=learner, C=C, n_layers=n_layers, **kwargs)


class SpatioTemporalClassifier(BaseClassifier):
    def __init__(self, input_length: int, input_channels: int, spatial_channels: int, dense: bool, C: int, n_layers: int, pool_level: int, spatial_kwargs: dict = {}, hidden_dimension: int = None, max_dilations_per_kernel: int = None, **kwargs) -> None:
        """First learn the relation between all input channels per timestep and
        then learn how this relationship over time relates to what class.

        Args:
            input_length (int): Largest expected input slice.
            This bounds the max dilution used. Longer inputs will still work.
            input_channels (int): How many channels/features.
            spatial_channels (int): How many output channels the spatial convolution generates after the first layer.
            dense (bool): Whether the spatial layer concatenates the input to the spatial output or not.
            C (int): How many classes.
            n_layers (int): How deep the classifier is.
            pool_level (int): How many pooling operations to combine. See model.blocks.PositivePooler.
            spatial_kwargs: Are passed to the SpatialConv.
            temporal_kwargs: Are passed to the TemporalRocket.
            kwargs: Are passed to the BaseClassifier.
        """
        if pool_level > 4:
            raise ValueError(f"{self.__class__.__name__} only supports a pool_level <= 4, is {pool_level}.")
        learner = nn.Sequential(
            SpatialConv(
                in_channels=input_channels,
                out_channels=spatial_channels,
                dense=dense,
                **spatial_kwargs,
            ),
            TemporalRocket(
                max_in_length=input_length,
                in_channels=spatial_channels + input_channels if dense else spatial_channels,
                hidden_dimension=hidden_dimension,
                max_dilations_per_kernel=max_dilations_per_kernel,
            ),
            PositivePooler(pool_level=pool_level),
        )
        shape = list(learner[-1].shape)
        shape[-1] = learner[-2].shape[-2]
        learner.shape = shape
        super().__init__(learner=learner, C=C, n_layers=n_layers, **kwargs)

