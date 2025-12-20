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

    cls_hidden_dims: Tuple[int, ...] = (1024, 1024, 512)

    train_epoch: int = 5
    lr: float = 1e-3
    batch_size: int = 128

    hidden_dim_drug: int = 128
    mpnn_hidden_size: int = 128
    mpnn_depth: int = 3

    cnn_target_filters: Tuple[int, ...] = (32, 64, 96)
    cnn_target_kernels: Tuple[int, ...] = (4, 8, 12)


def build_deeppurpose_model(cfg: Optional[HDNConfig] = None):
    """
    Tạo model DeepPurpose bằng dp_models.model_initialize(**config).
    Hàm trả về object của DeepPurpose; module PyTorch thật nằm ở thuộc tính `.model`.
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
    Giữ tên hàm như cũ để tương thích:
      seq_model = get_model().model
    """
    return build_deeppurpose_model(cfg)


@torch.no_grad()
def forward_logits(seq_torch_model: nn.Module, v_d, v_p) -> torch.Tensor:
    """
    Chuẩn hoá đầu ra về logits dạng (B,).
    DeepPurpose có thể trả (B,1), (B,) hoặc tuple/list.
    """
    out = seq_torch_model(v_d, v_p)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.view(-1)
