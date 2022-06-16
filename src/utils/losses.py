from typing import Dict

import torch
from torch.nn.modules.loss import _Loss


class DiceFocal(_Loss):
    def __init__(self, dice: _Loss, focal: _Loss, num_classes: float = None):
        super(DiceFocal, self).__init__()
        self.dice = dice
        self.focal = focal
        self.num_classes = num_classes

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(y_pred, y_true)
        focal_loss = self.focal(y_pred, y_true)
        if self.num_classes:
            focal_loss = focal_loss / self.num_classes
        dice_focal = dice_loss + focal_loss
        return {"loss": dice_focal, "dice_loss": dice_loss, "focal_loss": focal_loss}
