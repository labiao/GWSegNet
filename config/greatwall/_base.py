import os
from pathlib import Path

import torch
from catalyst import utils
from catalyst.contrib.nn import Lookahead
from torch.utils.data import DataLoader

from geoseg.datasets.great_wall_dataset import (
    CLASSES,
    GreatWallDataset,
    train_aug,
    val_aug,
)
from geoseg.losses import GWSegNetLoss
from geoseg.models.gwsegnet import GWSegNet


def _env_flag(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_config():
    experiment_name = "gwsegnet"
    region = os.getenv("GREAT_WALL_REGION", "beijing")
    data_root = Path(os.getenv(
        "GREAT_WALL_DATA_ROOT",
        f"./data/greatwall_{region}",
    ))
    dinov2_weight_path = os.getenv(
        "GWSEGNET_DINOV2_WEIGHTS",
        "./model_weights/dinov2_small.pth",
    )
    backbone_type = os.getenv("GWSEGNET_BACKBONE", "dinov2")
    backbone_name = os.getenv("GWSEGNET_BACKBONE_NAME") or None
    num_workers = int(os.getenv("GWSEGNET_NUM_WORKERS", "8"))

    max_epoch = 200
    train_batch_size = 8
    val_batch_size = 8
    test_batch_size = 2
    lr = 6e-4
    weight_decay = 0.01
    backbone_lr = 6e-6
    backbone_weight_decay = 0.01
    num_classes = len(CLASSES)
    ignore_index = num_classes

    weights_name = experiment_name
    weights_path = str(Path("model_weights") / "gwsegnet" / region / experiment_name)
    test_weights_name = weights_name
    log_name = f"greatwall/{region}/{experiment_name}"

    net = GWSegNet(
        num_classes=num_classes,
        pretrained=_env_flag("GWSEGNET_PRETRAINED", default=True),
        backbone_type=backbone_type,
        backbone_name=backbone_name,
        dinov2_weight_path=dinov2_weight_path,
    )
    loss = GWSegNetLoss(ignore_index=ignore_index)

    train_dataset = GreatWallDataset(
        data_root=data_root / "train",
        mode="train",
        mosaic_ratio=0.25,
        transform=train_aug,
    )
    val_dataset = GreatWallDataset(
        data_root=data_root / "test",
        mode="val",
        transform=val_aug,
    )
    evaluation_dataset = val_dataset
    test_dataset = GreatWallDataset(
        data_root=data_root / "test",
        mode="test",
        transform=val_aug,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    layerwise_params = {
        "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
    }
    net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
    base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
    optimizer = Lookahead(base_optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epoch,
    )

    return {
        "seed": 42,
        "max_epoch": max_epoch,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "test_batch_size": test_batch_size,
        "num_classes": num_classes,
        "classes": CLASSES,
        "eval_class_indices": [1],
        "weights_name": weights_name,
        "weights_path": weights_path,
        "test_weights_name": test_weights_name,
        "log_name": log_name,
        "monitor": "val_F1",
        "monitor_mode": "max",
        "save_top_k": 1,
        "save_last": True,
        "check_val_every_n_epoch": 1,
        "pretrained_ckpt_path": None,
        "resume_ckpt_path": None,
        "gpus": "auto",
        "accelerator": "auto",
        "strategy": "auto",
        "net": net,
        "loss": loss,
        "use_aux_loss": True,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "evaluation_dataset": evaluation_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
