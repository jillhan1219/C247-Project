# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, Sequence, Optional 

import numpy as np
import torch
import torchaudio
from torch import Tensor


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)


class RandomGain:
    """Multiply EMG by a random scalar gain sampled per example."""
    def __init__(self, min_gain: float = 0.8, max_gain: float = 1.2) -> None:
        assert 0 < min_gain <= max_gain
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)

    def __call__(self, x: Tensor) -> Tensor:
        # x: (..., channels) / (T, C) / (T, 16) etc.
        g = torch.empty((), device=x.device, dtype=x.dtype).uniform_(self.min_gain, self.max_gain)
        return x * g


class AdditiveGaussianNoise:
    """Add Gaussian noise to EMG. sigma can be absolute or relative to x.std()."""
    def __init__(self, sigma: float = 0.01, relative: bool = True, eps: float = 1e-6) -> None:
        self.sigma = float(sigma)
        self.relative = bool(relative)
        self.eps = float(eps)

    def __call__(self, x: Tensor) -> Tensor:
        if self.relative:
            scale = x.detach().std().clamp_min(self.eps)
            sigma = self.sigma * scale
        else:
            sigma = self.sigma
        noise = torch.randn_like(x) * sigma
        return x + noise


class RandomChannelDropout:
    """
    Randomly zero out electrode channels (along the last dimension).
    Works for shapes like (T, C) or (T, ..., C).
    """
    def __init__(self, p: float = 0.1, min_keep: int = 1) -> None:
        assert 0.0 <= p < 1.0
        self.p = float(p)
        self.min_keep = int(min_keep)

    def __call__(self, x: Tensor) -> Tensor:
        C = x.shape[-1]
        if C <= self.min_keep:
            return x
        # Decide which channels to keep
        keep = torch.rand(C, device=x.device) > self.p
        # Ensure at least min_keep channels kept
        if keep.sum().item() < self.min_keep:
            idx = torch.randperm(C, device=x.device)[: self.min_keep]
            keep = torch.zeros(C, device=x.device, dtype=torch.bool)
            keep[idx] = True
        mask = keep.to(dtype=x.dtype).view(*([1] * (x.ndim - 1)), C)  # broadcast
        return x * mask


class ChannelMask:
    """
    Keep only selected channels (along last dimension), zero out others.
    Deterministic given keep_indices.
    """
    def __init__(self, keep_indices: Sequence[int], num_channels: int = 16) -> None:
        self.keep_indices = list(map(int, keep_indices))
        self.num_channels = int(num_channels)

        assert len(self.keep_indices) > 0
        assert all(0 <= i < self.num_channels for i in self.keep_indices)

    def __call__(self, x: Tensor) -> Tensor:
        # x shape: (..., C) where C == num_channels
        C = x.shape[-1]
        assert C == self.num_channels, f"Expected last dim {self.num_channels}, got {C}"
        mask = torch.zeros(C, device=x.device, dtype=x.dtype)
        mask[self.keep_indices] = 1.0
        mask = mask.view(*([1] * (x.ndim - 1)), C)
        return x * mask