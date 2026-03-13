# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ===== æ–°å¢žï¼šSimple CNN =====
class CNNCTCModule(pl.LightningModule):

    # Level-1 CNN baseline:
    # - å¤ç”¨ SpectrogramNorm + MultiBandRotationInvariantMLP ä½œä¸ºå‰ç«¯ç‰¹å¾æå–
    # - ç”¨ç®€å•çš„ 1D Conv(æ²¿æ—¶é—´ç»´) æ›¿ä»£ TDSConvEncoder
    # - è¾“å‡ºä»ç„¶æ˜¯ (T, N, num_classes)ï¼Œç”¨ CTCLoss + åŒæ ·çš„ decoder/metrics

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_channels: int,
        kernel_size: int,
        num_cnn_layers: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # å¤ç”¨ baseline çš„å‰ç«¯ï¼šæ¯ä¸ª band è¾“å‡º mlp_features[-1]
        num_features = self.NUM_BANDS * mlp_features[-1]  # e.g., 2 * 384 = 768

        # ---- Front-end (same as baseline) ----
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # (T, N, num_features)
        )

        # ---- Temporal CNN encoder (Conv1d over time) ----
        layers: list[nn.Module] = []
        in_ch = num_features
        for i in range(num_cnn_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=cnn_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # keep time length
                )
            )
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = cnn_channels

        # project back to num_features so head stays simple
        layers.append(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=num_features,
                kernel_size=1,
                padding=0,
            )
        )
        self.temporal_cnn = nn.Sequential(*layers)

        # ---- CTC head ----
        self.classifier = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        # inputs: (T, N, bands=2, C=16, freq)
        # returns: (T, N, num_classes)

        x = self.frontend(inputs)          # (T, N, F)
        x = x.permute(1, 2, 0)             # (N, F, T) for Conv1d
        x = self.temporal_cnn(x)           # (N, F, T) (same T length)
        x = x.permute(2, 0, 1)             # (T, N, F)
        emissions = self.classifier(x)     # (T, N, num_classes)
        return emissions

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # æˆ‘ä»¬çš„ Conv1d ç”¨äº† padding ä¿æŒæ—¶é—´é•¿åº¦ => emissions.shape[0] == inputs.shape[0]
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ===== Level-2: Stronger CNN (Residual + BN + Dilated Conv) =====

class _ResidualTCNBlock(nn.Module):
    # è¾“å…¥/è¾“å‡º: (N, C, T)
    # ç›®çš„ï¼šæ›´æ·±çš„ temporal encoderï¼Œä½†ä¿æŒ T ä¸å˜ï¼ˆé€‚é… CTCï¼‰
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size // 2) * dilation  # ä¿æŒé•¿åº¦ä¸å˜

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)  # residual


class CNNCTCModuleV2(pl.LightningModule):

    # Level-2 (strong):
    # - å‰ç«¯ä»ç„¶å¤ç”¨ baselineï¼šSpectrogramNorm + MultiBandRotationInvariantMLP
    # - Temporal encoder æ¢æˆ Residual Dilated CNNï¼ˆTCN styleï¼‰
    # - è¾“å‡ºä»æ˜¯ (T, N, num_classes)ï¼ŒCTC loss ä¸å˜


    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_channels: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
        dilation_growth: int,   # e.g. 2 -> 1,2,4,8...
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # å‰ç«¯è¾“å‡ºç»´åº¦
        num_features = self.NUM_BANDS * mlp_features[-1]  # e.g., 2 * 384 = 768

        # ---- Frontend (same as baseline) ----
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # (T, N, num_features)
        )

        # ---- Project to cnn_channels ----
        self.in_proj = nn.Conv1d(num_features, cnn_channels, kernel_size=1, bias=False)

        # ---- Residual Dilated TCN blocks ----
        blocks = []
        dilation = 1
        for _ in range(num_blocks):
            blocks.append(_ResidualTCNBlock(cnn_channels, kernel_size, dilation, dropout))
            dilation *= dilation_growth
        self.tcn = nn.Sequential(*blocks)

        # ---- Project back to num_features ----
        self.out_proj = nn.Conv1d(cnn_channels, num_features, kernel_size=1, bias=False)

        # ---- CTC head ----
        self.classifier = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {f"{phase}_metrics": metrics.clone(prefix=f"{phase}/") for phase in ["train", "val", "test"]}
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C=16, freq)
        x = self.frontend(inputs)       # (T, N, F)
        x = x.permute(1, 2, 0)          # (N, F, T)
        x = self.in_proj(x)             # (N, C, T)
        x = self.tcn(x)                 # (N, C, T)
        x = self.out_proj(x)            # (N, F, T)
        x = x.permute(2, 0, 1)          # (T, N, F)
        return self.classifier(x)       # (T, N, num_classes)

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)
        emission_lengths = input_lengths  # ä¿æŒ T ä¸å˜ï¼ˆstride=1 + paddingï¼‰

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ===== Level-3: Strong CNN (PreNorm + GroupNorm + DepthwiseSeparable + GLU + optional SE) =====

class _SE1D(nn.Module):
    # x: (N, C, T)
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.net(self.pool(x))  # (N, C, 1)
        return x * gate


class _DWSeparableGLUBlock(nn.Module):
    # PreNorm residual block for (N, C, T)
    # ç»“æž„ï¼šx + Drop( PW( GLU( PW2( DW( Norm(x) ) ) ) ) ) (+ optional SE)
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_se: bool,
        se_ratio: float,
    ):
        super().__init__()
        pad = (kernel_size // 2) * dilation  # keep T

        self.norm = nn.GroupNorm(1, channels)  # æ¯” BN æ›´ç¨³ï¼ˆåºåˆ— + augmentationï¼‰
        self.dw = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation,
            groups=channels, bias=False
        )
        # pointwise -> 2C for GLU
        self.pw1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.use_se = use_se
        self.se = _SE1D(channels, se_ratio) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dw(y)
        y = self.pw1(y)                  # (N, 2C, T)
        y = nn.functional.glu(y, dim=1)  # (N, C, T)  é—¨æŽ§
        y = self.drop(y)
        y = self.pw2(y)
        y = self.se(y)
        y = self.drop(y)
        return x + y


class CNNCTCModuleV3(pl.LightningModule):
    # Level-3 (stronger):
    # - å‰ç«¯ï¼šSpectrogramNorm + MultiBandRotationInvariantMLPï¼ˆä¸åŠ¨ï¼Œä¿è¯å¯¹æ¯”å…¬å¹³ï¼‰
    # - Temporal encoderï¼šPreNorm + GroupNorm + DepthwiseSeparable + GLU + Dilated Residual blocks
    # - ä¿æŒ T ä¸å˜ => emission_lengths = input_lengths

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_channels: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
        dilation_growth: int,
        use_se: bool,
        se_ratio: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]  # e.g., 2*384 = 768

        # ---- Frontend (same as baseline) ----
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # (T, N, F)
        )

        # ---- Project to cnn_channels ----
        self.in_proj = nn.Conv1d(num_features, cnn_channels, kernel_size=1, bias=False)

        # ---- Strong temporal encoder ----
        blocks: list[nn.Module] = []
        dilation = 1
        for _ in range(num_blocks):
            blocks.append(
                _DWSeparableGLUBlock(
                    channels=cnn_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_se=use_se,
                    se_ratio=se_ratio,
                )
            )
            dilation *= dilation_growth
        self.encoder = nn.Sequential(*blocks)

        # ---- Back to num_features ----
        self.out_proj = nn.Conv1d(cnn_channels, num_features, kernel_size=1, bias=False)

        # ---- CTC head ----
        self.classifier = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {f"{phase}_metrics": metrics.clone(prefix=f"{phase}/") for phase in ["train", "val", "test"]}
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)   # (T, N, F)
        x = x.permute(1, 2, 0)      # (N, F, T)
        x = self.in_proj(x)         # (N, C, T)
        x = self.encoder(x)         # (N, C, T)
        x = self.out_proj(x)        # (N, F, T)
        x = x.permute(2, 0, 1)      # (T, N, F)
        return self.classifier(x)   # (T, N, num_classes)

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)
        emission_lengths = input_lengths  # T ä¸å˜

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ===== Level-3 fix: CNNCTCModuleV3_1 (supports dilation_cycle) =====

class _ResGNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, gn_groups: int):
        super().__init__()
        pad = (kernel_size // 2) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class CNNCTCModuleV3_1(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_channels: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
        dilation_cycle: Sequence[int],   # âœ… ä¸Ž yaml å¯¹é½
        gn_groups: int,                 # âœ… ä¸Ž yaml å¯¹é½
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # (T,N,F)
        )

        self.in_proj = nn.Conv1d(num_features, cnn_channels, kernel_size=1, bias=False)

        cycle = list(dilation_cycle)
        blocks = []
        for i in range(num_blocks):
            d = cycle[i % len(cycle)]
            blocks.append(_ResGNBlock(cnn_channels, kernel_size, d, dropout, gn_groups))
        self.tcn = nn.Sequential(*blocks)

        self.out_proj = nn.Conv1d(cnn_channels, num_features, kernel_size=1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {f"{phase}_metrics": metrics.clone(prefix=f"{phase}/") for phase in ["train", "val", "test"]}
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)   # (T,N,F)
        x = x.permute(1,2,0)        # (N,F,T)
        x = self.in_proj(x)         # (N,C,T)
        x = self.tcn(x)             # (N,C,T)
        x = self.out_proj(x)        # (N,F,T)
        x = x.permute(2,0,1)        # (T,N,F)
        return self.classifier(x)   # (T,N,classes)

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ===== Level-3: Level2 + SE-Gating (stable improvement) =====

class _SE1D(nn.Module):
    # è¾“å…¥: (N, C, T) -> è¾“å‡º: (N, C, T)
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        g = self.pool(x)      # (N,C,1)
        s = self.net(g)       # (N,C,1)
        return x * s          # channel-wise gate


class _ResidualTCNBlockV2SE(nn.Module):
    # Pre-Act style: BN -> GELU -> Conv, æ›´ç¨³
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, se_reduction: int):
        super().__init__()
        pad = (kernel_size // 2) * dilation

        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False)

        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.se = _SE1D(channels, reduction=se_reduction)

    def forward(self, x):
        y = self.conv1(self.drop(self.act(self.bn1(x))))
        y = self.conv2(self.drop(self.act(self.bn2(y))))
        y = self.se(y)
        return x + y


class CNNCTCModuleV2_SE(pl.LightningModule):
    # åŸºäºŽ Level2ï¼šä¿æŒ BN + dilation_growth
    # å¢žå¼ºç‚¹ï¼šResidual block åŠ  SE-gating + Pre-Actï¼Œé€šå¸¸æ›´ç¨³ã€æ›´å¼º
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_channels: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
        dilation_growth: int,
        se_reduction: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # (T,N,F)
        )

        self.in_proj = nn.Conv1d(num_features, cnn_channels, kernel_size=1, bias=False)

        blocks = []
        dilation = 1
        for _ in range(num_blocks):
            blocks.append(_ResidualTCNBlockV2SE(cnn_channels, kernel_size, dilation, dropout, se_reduction))
            dilation *= dilation_growth
        self.tcn = nn.Sequential(*blocks)

        self.out_proj = nn.Conv1d(cnn_channels, num_features, kernel_size=1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {f"{phase}_metrics": metrics.clone(prefix=f"{phase}/") for phase in ["train", "val", "test"]}
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)   # (T,N,F)
        x = x.permute(1,2,0)        # (N,F,T)
        x = self.in_proj(x)         # (N,C,T)
        x = self.tcn(x)             # (N,C,T)
        x = self.out_proj(x)        # (N,F,T)
        x = x.permute(2,0,1)        # (T,N,F)
        return self.classifier(x)

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )