from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch


class Multiclass_IoU_Dice(smp.utils.base.Metric):
    def __init__(
        self,
        mode="iou",
        threshold=None,
        eps=1e-7,
        nan_score_on_empty=False,
        classes_of_interest=None,
        ignore_index=None,
        mean_score=True,
        name=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.mode = mode
        self.threshold = threshold
        self.eps = eps
        self.nan_score_on_empty = nan_score_on_empty
        self.classes_of_interest = classes_of_interest
        self.ignore_index = ignore_index
        self.mean_score = mean_score
        self._name = name

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        score = multiclass_dice_iou_score(
            y_pred,
            y_true,
            self.mode,
            self.threshold,
            self.eps,
            self.nan_score_on_empty,
            self.classes_of_interest,
            self.ignore_index,
            self.mean_score,
        )

        return score


def binary_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold: Optional[float] = None,
    nan_score_on_empty=False,
    eps: float = 1e-7,
    ignore_index=None,
) -> float:
    """Compute IoU score between two image tensors"""
    assert mode in {"dice", "iou"}

    # Make binary predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    if ignore_index is not None:
        mask = (y_true != ignore_index).to(y_true.dtype)
        y_true = y_true * mask
        y_pred = y_pred * mask

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if mode == "dice":
        score = (2.0 * intersection) / (cardinality + eps)
    else:
        score = intersection / (cardinality - intersection + eps)

    has_targets = torch.sum(y_true) > 0
    has_predicted = torch.sum(y_pred) > 0

    if not has_targets:
        if nan_score_on_empty:
            score = np.nan
        else:
            score = float(not has_predicted)
    return score


def multiclass_dice_iou_score(
    y_preds: torch.Tensor,
    y_trues: torch.Tensor,
    mode="iou",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
    ignore_index=None,
    mean_score=True,
):
    batch_size = y_preds.shape[0]
    mious = []

    for k in range(batch_size):
        y_pred = y_preds[k]
        y_true = y_trues[k]

        ious = []
        num_classes = y_pred.size(0)
        y_pred = y_pred.argmax(dim=0)

        if classes_of_interest is None:
            classes_of_interest = range(num_classes)

        for class_index in classes_of_interest:
            y_pred_i = (y_pred == class_index).float()
            y_true_i = (y_true == class_index).float()

            if ignore_index is not None:
                not_ignore_mask = (y_true != ignore_index).float()
                y_pred_i *= not_ignore_mask
                y_true_i *= not_ignore_mask

            iou = binary_dice_iou_score(
                y_pred=y_pred_i,
                y_true=y_true_i,
                mode=mode,
                nan_score_on_empty=nan_score_on_empty,
                threshold=threshold,
                eps=eps,
            )
            ious.append(iou)
        mious.append(ious)

    if mean_score:
        return np.nanmean(np.nanmean(mious, axis=1))
    else:
        return mious
