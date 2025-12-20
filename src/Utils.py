# src/Utils.py
import csv
import os
import random
import numpy as np
import torch
from loguru import logger

NEG_LABEL = 0
POS_LABEL = 1


def check_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"create dir: {path}")
    else:
        logger.info(f"dir exists: {path}")

def csv_record(path: str, data: dict):
    all_header = [
        "epoch","batch","lr","loss","avg_loss","epoch_loss",
        "auprc","auroc","accuracy","f1","precision","recall",
        "sensitivity","specificity"
    ]
    row, header = [], []
    for k in all_header:
        if k in data:
            header.append(k)
            row.append(data[k])

    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

def setup_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(
        "seed=%d | random=%.6f | torch=%.6f | np=%.6f",
        seed, random.random(), float(torch.rand(1)), float(np.random.rand(1))
    )

def class_metrics(y_true, y_pred):
    """
    y_true, y_pred: numpy array of 0/1 labels
    POS_LABEL=0, NEG_LABEL=1 (the convention you used)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    TP = np.sum((y_pred == POS_LABEL) & (y_true == POS_LABEL))
    TN = np.sum((y_pred == NEG_LABEL) & (y_true == NEG_LABEL))
    FP = np.sum((y_pred == POS_LABEL) & (y_true == NEG_LABEL))
    FN = np.sum((y_pred == NEG_LABEL) & (y_true == POS_LABEL))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 1.0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall      = sensitivity
    accuracy    = (TP + TN) / max(1, (TP + TN + FP + FN))
    f1          = (2*TP) / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 1.0

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }
