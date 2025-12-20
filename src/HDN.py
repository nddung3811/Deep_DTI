from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from DeepPurpose import utils as dp_utils
from DeepPurpose import DTI as dp_models


@dataclass
class HDNConfig:
    drug_encoding: str = "MPNN"
    target_encoding: str = "CNN"

    # classifier head of DeepPurpose
    cls_hidden_dims: Tuple[int, ...] = (1024, 1024, 512)

    # DeepPurpose training config fields (even if we train outside)
    train_epoch: int = 5
    lr: float = 1e-3
    batch_size: int = 128

    # drug encoder params
    hidden_dim_drug: int = 128
    mpnn_hidden_size: int = 128
    mpnn_depth: int = 3

    # protein encoder params
    cnn_target_filters: Tuple[int, ...] = (32, 64, 96)
    cnn_target_kernels: Tuple[int, ...] = (4, 8, 12)


def build_deeppurpose_model(cfg: Optional[HDNConfig] = None):
    """
    Returns DeepPurpose 'model' object created by dp_models.model_initialize(**config).
    NOTE: that returned object has `.model` inside (the actual torch module).
    """
    cfg = cfg or HDNConfig()
    config = dp_utils.generate_config(
        drug_encoding=cfg.drug_encoding,
        target_encoding=cfg.target_encoding,
        cls_hidden_dims=list(cfg.cls_hidden_dims),
        train_epoch=cfg.train_epoch,
        LR=cfg.lr,
        batch_size=cfg.batch_size,
        hidden_dim_drug=cfg.hidden_dim_drug,
        mpnn_hidden_size=cfg.mpnn_hidden_size,
        mpnn_depth=cfg.mpnn_depth,
        cnn_target_filters=list(cfg.cnn_target_filters),
        cnn_target_kernels=list(cfg.cnn_target_kernels),
    )
    return dp_models.model_initialize(**config)


def get_model(cfg: Optional[HDNConfig] = None):
    """
    Backward-compatible name you were using:
      seq_model = get_model().model
    """
    return build_deeppurpose_model(cfg)


@torch.no_grad()
def forward_logits(seq_torch_model: nn.Module, v_d, v_p) -> torch.Tensor:
    """
    Unify shape:
      returns (B,) logits.
    DeepPurpose model usually returns shape (B,1) or (B,).
    """
    out = seq_torch_model(v_d, v_p)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = out.view(-1)
    return out
