# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding module for Transformer.
    
    Args:
        d_model: the number of expected features in the input
        max_len: the maximum length of the input sequence (default: 5000)
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Precompute positional encoding for max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor of shape (seq_len, batch_size, d_model)
        """
        seq_len = x.size(0)
        
        if seq_len <= self.max_len:
            return x + self.pe[:seq_len, :]
        else:
            # Compute positional encoding on the fly for longer sequences
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * 
                               (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(seq_len, 1, self.d_model, device=x.device)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            return x + pe


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with self-attention and feed-forward layers.
    
    Args:
        d_model: the number of expected features in the input
        nhead: the number of heads in the multiheadattention models
        dim_feedforward: the dimension of the feedforward network model
        dropout: the dropout value
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor of shape (seq_len, batch_size, d_model)
            src_mask: Tensor of shape (seq_len, seq_len)
        Returns:
            Tensor of shape (seq_len, batch_size, d_model)
        """
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    """Transformer encoder composed of multiple encoder blocks.

    Args:
        d_model: the number of expected features in the input
        nhead: the number of heads in the multiheadattention models
        num_layers: the number of encoder layers
        dim_feedforward: the dimension of the feedforward network model
        dropout: the dropout value
        max_len: the maximum length of the input sequence
        chunk_size: if > 0, long sequences are split into non-overlapping chunks
            of this size and each chunk is processed independently through the
            attention layers. Avoids OOM when running full sessions at test time.
            Set to 0 to disable chunking (default).
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, max_len: int = 5000,
                 chunk_size: int = 0):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model
        self.chunk_size = chunk_size

    def _encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.transformer_encoder:
            src = layer(src, src_mask)
        return src

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor of shape (seq_len, batch_size, d_model)
            src_mask: Tensor of shape (seq_len, seq_len)
        Returns:
            Tensor of shape (seq_len, batch_size, d_model)
        """
        # Apply positional encoding globally so positions are correct across chunks
        src = self.pos_encoder(src)

        if self.chunk_size > 0 and src.size(0) > self.chunk_size:
            # Split into chunks along time dim, encode each independently, then concat
            chunks = src.split(self.chunk_size, dim=0)
            return torch.cat([self._encode(chunk) for chunk in chunks], dim=0)

        return self._encode(src, src_mask)


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    Supports two normalization types:
    - BatchNorm2d: Standard batch normalization (default)
    - RollingTimeNorm: Causal rolling normalization along time axis

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
        spec_norm_type (str): Type of normalization - 'BatchNorm2d' or 'RollingTimeNorm'
    """

    def __init__(self, channels: int, spec_norm_type: str = "BatchNorm2d") -> None:
        super().__init__()
        self.channels = channels
        self.spec_norm_type = spec_norm_type

        if self.spec_norm_type == "RollingTimeNorm":
            self.spec_norm = RollingTimeNorm(eps=0.1)
        elif self.spec_norm_type == "BatchNorm2d":
            self.batch_norm = nn.BatchNorm2d(channels)
        elif self.spec_norm_type == "None":
            self.spec_norm = None
        else:
            raise ValueError(f"Invalid spec_norm_type, got {spec_norm_type}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape

        if self.spec_norm_type == "RollingTimeNorm":
            return self.spec_norm(inputs)

        if self.spec_norm_type == "BatchNorm2d":
            x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
            x = x.reshape(N, bands * C, freq, T)
            x = self.batch_norm(x)
            x = x.reshape(N, bands, C, freq, T)
            return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)

        # spec_norm_type == "None"
        return inputs


class RollingTimeNorm(nn.Module):
    r"""Causally normalize a 5D tensor (T, N, bands, channels, freq) along the time axis.

    For each sample (N) and for every (band, channel, frequency) location, this module
    normalizes the data as follows:

    - For the first `warmup` time steps (default 125), the statistics (mean and standard deviation)
      are computed over the entire warmup period (i.e. time indices 0 to 124) and are used for
      every time step in that period.
    - For any subsequent time step t >= warmup, the statistics are computed over
      all time steps from 0 up through t.

    Args:
        warmup (int): Number of time bins to use for warmup (default: 125).
        eps (float): A small constant added for numerical stability.
    """

    def __init__(self, warmup: int = 125, eps: float = 1e-5):
        super().__init__()
        self.warmup = warmup
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (T, N, bands, channels, freq).

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        T, N, bands, channels, freq = x.shape
        device, dtype = x.device, x.dtype

        cumsum = x.cumsum(dim=0)
        cumsum2 = (x**2).cumsum(dim=0)

        time_range = torch.arange(1, T + 1, device=device, dtype=dtype).view(T, 1, 1, 1, 1)

        cum_mean = cumsum / time_range
        cum_mean2 = cumsum2 / time_range
        cum_std = torch.sqrt(cum_mean2 - cum_mean**2 + self.eps)

        if T >= self.warmup:
            warmup_mean = cum_mean[self.warmup - 1]
            warmup_mean2 = cum_mean2[self.warmup - 1]
            warmup_std = torch.sqrt(warmup_mean2 - warmup_mean**2 + self.eps)

            cum_mean[: self.warmup] = warmup_mean.unsqueeze(0).expand(
                self.warmup, N, bands, channels, freq
            )
            cum_std[: self.warmup] = warmup_std.unsqueeze(0).expand(
                self.warmup, N, bands, channels, freq
            )
        else:
            warmup_mean = cum_mean[-1]
            warmup_std = torch.sqrt(cum_mean2[-1] - warmup_mean**2 + self.eps)
            cum_mean[:] = warmup_mean
            cum_std[:] = warmup_std

        return (x - cum_mean) / cum_std


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
        share_hand_weights (bool): If True, use a single shared MLP for all
            bands instead of separate MLPs. (default: False)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
        share_hand_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim
        self.share_hand_weights = share_hand_weights

        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(1 if share_hand_weights else num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        if self.share_hand_weights:
            outputs_per_band = [
                self.mlps[0](_input) for _input in inputs_per_band
            ]
        else:
            outputs_per_band = [
                mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
            ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
        share_hand_weights (bool): If True, process left/right halves of the
            feature dimension in parallel with shared conv and norm weights.
    """

    def __init__(self, channels: int, width: int, kernel_width: int,
                 share_hand_weights: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.share_hand_weights = share_hand_weights

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        norm_features = channels * (width // 2 if share_hand_weights else width)
        self.layer_norm = nn.LayerNorm(norm_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        if self.share_hand_weights:
            half_C = C // 2
            half_width = self.width // 2
            # Process both halves in parallel: (T, N, C) -> (T, 2N, C/2)
            x = inputs.view(T_in, N * 2, half_C)

            x = x.movedim(0, -1).reshape(N * 2, self.channels, half_width, T_in)
            x = self.conv2d(x)
            x = self.relu(x)
            x = x.reshape(N * 2, half_C, -1).movedim(-1, 0)  # (T_out, 2N, C/2)

            T_out = x.shape[0]
            x = x + inputs.view(T_in, N * 2, half_C)[-T_out:]
            x = self.layer_norm(x)
            return x.view(T_out, N, C)
        else:
            # TNC -> NCT -> NcwT
            x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
            x = self.conv2d(x)
            x = self.relu(x)
            x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

            T_out = x.shape[0]
            x = x + inputs[-T_out:]
            return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        share_hand_weights (bool): If True, process left/right halves of the
            feature dimension in parallel with shared FC and norm weights.
    """

    def __init__(self, num_features: int, share_hand_weights: bool = False) -> None:
        super().__init__()
        self.share_hand_weights = share_hand_weights

        fc_features = num_features // 2 if share_hand_weights else num_features
        self.fc_block = nn.Sequential(
            nn.Linear(fc_features, fc_features),
            nn.ReLU(),
            nn.Linear(fc_features, fc_features),
        )
        self.layer_norm = nn.LayerNorm(fc_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.share_hand_weights:
            T, N, C = inputs.shape
            half_C = C // 2
            x = inputs.view(T, N * 2, half_C)
            x = self.fc_block(x) + x
            x = self.layer_norm(x)
            return x.view(T, N, C)
        else:
            x = self.fc_block(inputs)
            x = x + inputs
            return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
        share_hand_weights (bool): If True, share conv/FC weights between
            left and right halves of the feature dimension.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        share_hand_weights: bool = False,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width,
                                   share_hand_weights=share_hand_weights),
                    TDSFullyConnectedBlock(num_features,
                                           share_hand_weights=share_hand_weights),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)
