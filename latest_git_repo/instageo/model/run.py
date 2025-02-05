# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Run Module Containing Training, Evaluation and Inference Logic."""

import json
import logging
import os
from functools import partial
from typing import Any, List, Optional, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from instageo.model.dataloader import (
    InstaGeoDataset,
    process_and_augment,
    process_data,
    process_test,
)
from instageo.model.infer_utils import chip_inference, sliding_window_inference
# --------------------------------------------------------------------------
# CHANGED: Import our new CoAtNet-based segmentation model (instead of PrithviSeg)
# from instageo.model.model import PrithviSeg  # <- remove or comment out
from instageo.model.model import CoAtNetSeg  # <- use this instead
# --------------------------------------------------------------------------


pl.seed_everything(seed=1042, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def check_required_flags(required_flags: List[str], config: DictConfig) -> None:
    ...
    # same as before


def get_device() -> str:
    ...
    # same as before


def eval_collate_fn(batch: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    # same as before


def infer_collate_fn(batch: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    # same as before


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 1,
    collate_fn: Optional = None,
    pin_memory: bool = True,
) -> DataLoader:
    ...
    # same as before


class CoAtNetSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper around the CoAtNetSeg model
    (similar to the old PrithviSegmentationModule but referencing CoAtNetSeg).
    """

    def __init__(
        self,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        num_classes: int = 2,
        temporal_step: int = 1,
        class_weights: List[float] = None,
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
        coatnet_variant: str = "coatnet_0",
        in_chans: int = 3,
    ):
        """
        Args:
            image_size (int): Size of input image (HxW).
            num_classes (int): Number of segmentation classes.
            freeze_backbone (bool): Freeze CoAtNet backbone weights if True.
            temporal_step (int): You can pass this if you do multi-temporal;
                                 might be used to set in_chans, etc.
            coatnet_variant (str): Which coatnet variant to load.
            in_chans (int): Number of input channels (including multi-temporal).
            ...
        """
        super().__init__()
        # If you have multi-temporal data, you might set in_chans = (temporal_step * #bands).
        # Or the caller can do that. For example:
        #   in_chans = len(cfg.dataloader.bands) * temporal_step
        # We'll assume the user does that logic outside.

        self.save_hyperparameters()

        self.net = CoAtNetSeg(
            image_size=image_size,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            in_chans=in_chans,
            coatnet_variant=coatnet_variant,
        )

        weight_tensor = torch.tensor(class_weights).float() if class_weights else None
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weight_tensor
        )
        self.learning_rate = learning_rate
        self.ignore_index = ignore_index
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels.long())
        self.log_metrics(outputs, labels, "train", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels.long())
        self.log_metrics(outputs, labels, "val", loss)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels.long())
        self.log_metrics(outputs, labels, "test", loss)
        return loss

    def predict_step(self, batch: Any) -> torch.Tensor:
        prediction = self.forward(batch)
        probabilities = torch.nn.functional.softmax(prediction, dim=1)[:, 1, :, :]
        return probabilities

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0
        )
        return [optimizer], [scheduler]

    def log_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        stage: str,
        loss: torch.Tensor,
    ) -> None:
        out = self.compute_metrics(predictions, labels)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_aAcc", out["acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mIoU", out["iou"], on_step=True, on_epoch=True, prog_bar=True)

        for idx, value in enumerate(out["iou_per_class"]):
            self.log(f"{stage}_IoU_{idx}", value, on_step=True, on_epoch=True)
        for idx, value in enumerate(out["acc_per_class"]):
            self.log(f"{stage}_Acc_{idx}", value, on_step=True, on_epoch=True)
        for idx, value in enumerate(out["precision_per_class"]):
            self.log(f"{stage}_Precision_{idx}", value, on_step=True, on_epoch=True)
        for idx, value in enumerate(out["recall_per_class"]):
            self.log(f"{stage}_Recall_{idx}", value, on_step=True, on_epoch=True)

    def compute_metrics(
        self, pred_mask: torch.Tensor, gt_mask: torch.Tensor
    ) -> dict:
        pred_mask = torch.argmax(pred_mask, dim=1)
        no_ignore = gt_mask.ne(self.ignore_index).to(self.device)
        pred_mask = pred_mask.masked_select(no_ignore).cpu().numpy()
        gt_mask = gt_mask.masked_select(no_ignore).cpu().numpy()
        classes = np.unique(np.concatenate((gt_mask, pred_mask)))

        iou_per_class = []
        accuracy_per_class = []
        precision_per_class = []
        recall_per_class = []

        for clas in classes:
            pred_cls = pred_mask == clas
            gt_cls = gt_mask == clas

            intersection = np.logical_and(pred_cls, gt_cls)
            union = np.logical_or(pred_cls, gt_cls)
            true_positive = np.sum(intersection)
            false_positive = np.sum(pred_cls) - true_positive
            false_negative = np.sum(gt_cls) - true_positive

            if np.any(union):
                iou = np.sum(intersection) / np.sum(union)
                iou_per_class.append(iou)

            accuracy = true_positive / np.sum(gt_cls) if np.sum(gt_cls) > 0 else 0
            accuracy_per_class.append(accuracy)

            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) > 0
                else 0
            )
            precision_per_class.append(precision)

            recall = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) > 0
                else 0
            )
            recall_per_class.append(recall)

        mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0
        overall_accuracy = np.sum(pred_mask == gt_mask) / gt_mask.size

        return {
            "iou": mean_iou,
            "acc": overall_accuracy,
            "acc_per_class": accuracy_per_class,
            "iou_per_class": iou_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
        }


def compute_mean_std(data_loader: DataLoader) -> Tuple[List[float], List[float]]:
    ...
    # same as before


@hydra.main(config_path="configs", version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runner Entry Point."""
    log.info(f"Script: {__file__}")
    log.info(f"Imported hydra config:\n{OmegaConf.to_yaml(cfg)}")

    BANDS = cfg.dataloader.bands
    MEAN = cfg.dataloader.mean
    STD = cfg.dataloader.std
    IM_SIZE = cfg.dataloader.img_size
    TEMPORAL_SIZE = cfg.dataloader.temporal_dim

    batch_size = cfg.train.batch_size
    root_dir = cfg.root_dir
    valid_filepath = cfg.valid_filepath
    train_filepath = cfg.train_filepath
    test_filepath = cfg.test_filepath
    checkpoint_path = cfg.checkpoint_path

    if cfg.mode == "stats":
        ...
        # same as before

    if cfg.mode == "train":
        ...
        # Create train_dataset, valid_dataset as before
        train_loader = create_dataloader(train_dataset, ...)
        valid_loader = create_dataloader(valid_dataset, ...)

        # ---------------------------------------------------------------------
        # CHANGED: Use CoAtNetSegmentationModule instead of Prithvi
        model = CoAtNetSegmentationModule(
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
            coatnet_variant=cfg.model.coatnet_variant,
            in_chans=len(BANDS) * cfg.dataloader.temporal_dim,
        )
        # ---------------------------------------------------------------------

        hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        checkpoint_callback = ModelCheckpoint(
            monitor="val_mIoU",
            dirpath=hydra_out_dir,
            filename="instageo_epoch-{epoch:02d}-val_iou-{val_mIoU:.2f}",
            auto_insert_metric_name=False,
            mode="max",
            save_top_k=3,
        )
        logger = TensorBoardLogger(hydra_out_dir, name="instageo")

        trainer = pl.Trainer(
            accelerator=get_device(),
            max_epochs=cfg.train.num_epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
        )
        trainer.fit(model, train_loader, valid_loader)

    elif cfg.mode == "eval":
        ...
        # CHANGED: Load CoAtNetSegmentationModule
        model = CoAtNetSegmentationModule.load_from_checkpoint(
            checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
            coatnet_variant=cfg.model.coatnet_variant,
            in_chans=len(BANDS) * cfg.dataloader.temporal_dim,
        )
        trainer = pl.Trainer(accelerator=get_device())
        result = trainer.test(model, dataloaders=test_loader)
        log.info(f"Evaluation results:\n{result}")

    elif cfg.mode in ["sliding_inference", "chip_inference"]:
        # same logic, just change the class that is loaded
        ...
        model = CoAtNetSegmentationModule.load_from_checkpoint(
            checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
            coatnet_variant=cfg.model.coatnet_variant,
            in_chans=len(BANDS) * cfg.dataloader.temporal_dim,
        )
        # Then do the inference logic as before...
        ...

if __name__ == "__main__":
    main()