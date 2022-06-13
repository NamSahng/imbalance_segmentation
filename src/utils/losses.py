import torch
from typing import Dict
from torch.nn.modules.loss import _Loss

class DiceFocal(_Loss):
    def __init__(self, dice: _Loss, focal: _Loss):
        super(DiceFocal, self).__init__()
        self.dice = dice
        self.focal = focal
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(y_pred, y_true)
        focal_loss = self.focal(y_pred, y_true)
        dice_focal =dice_loss + focal_loss
        return {'loss': dice_focal, 'dice_loss': dice_loss, 'focal_loss': focal_loss}
