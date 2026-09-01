import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

from tools.cfg import py2cfg
from tools.metric import Evaluator


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to a Python config.")
    return parser.parse_args()


class SegmentationModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.train_metrics = Evaluator(num_class=config.num_classes)
        self.val_metrics = Evaluator(num_class=config.num_classes)

    def forward(self, image):
        return self.net(image)

    @staticmethod
    def _main_logits(prediction):
        return prediction[0] if isinstance(prediction, (list, tuple)) else prediction

    @staticmethod
    def _update_metrics(evaluator, mask, logits):
        predicted_mask = nn.functional.softmax(logits, dim=1).argmax(dim=1)
        for target, prediction in zip(mask, predicted_mask):
            evaluator.add_batch(target.detach().cpu().numpy(), prediction.detach().cpu().numpy())

    def training_step(self, batch, batch_idx):
        prediction = self.net(batch["img"])
        loss = self.loss(prediction, batch["gt_semantic_seg"])
        self._update_metrics(
            self.train_metrics,
            batch["gt_semantic_seg"],
            self._main_logits(prediction),
        )
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self(batch["img"])
        loss = self.loss(prediction, batch["gt_semantic_seg"])
        self._update_metrics(
            self.val_metrics,
            batch["gt_semantic_seg"],
            self._main_logits(prediction),
        )
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _log_epoch_metrics(self, split, evaluator):
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        class_indices = list(getattr(
            self.config,
            "eval_class_indices",
            range(self.config.num_classes),
        ))
        if not class_indices:
            raise ValueError("eval_class_indices must contain at least one class index.")

        mean_iou = float(np.nanmean(iou_per_class[class_indices]))
        mean_f1 = float(np.nanmean(f1_per_class[class_indices]))
        overall_accuracy = float(evaluator.OA())
        per_class = {
            class_name: float(iou)
            for class_name, iou in zip(self.config.classes, iou_per_class)
        }
        print(f"{split}: IoU={mean_iou:.4f}, F1={mean_f1:.4f}, per-class IoU={per_class}")
        evaluator.reset()

        self.log_dict(
            {
                f"{split}_mIoU": mean_iou,
                f"{split}_F1": mean_f1,
                f"{split}_OA": overall_accuracy,
            },
            prog_bar=True,
            sync_dist=True,
        )

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train", self.train_metrics)

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val", self.val_metrics)

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


def main():
    args = parse_args()
    config = py2cfg(args.config)
    seed_everything(getattr(config, "seed", 42))

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name,
    )
    logger = CSVLogger("lightning_logs", name=config.log_name)

    model = SegmentationModule(config)
    if config.pretrained_ckpt_path:
        model = SegmentationModule.load_from_checkpoint(
            config.pretrained_ckpt_path,
            config=config,
        )

    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator=getattr(config, "accelerator", "auto"),
        strategy=getattr(config, "strategy", "auto"),
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        logger=logger,
        num_sanity_val_steps=getattr(config, "num_sanity_val_steps", 2),
    )
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
    main()
