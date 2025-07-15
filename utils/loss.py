import torch
import torch.nn as nn

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

         # ========== Dice Loss ==========
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        # ========== Focal Loss ==========
        # Calculate probabilities
        #p = torch.sigmoid(pred_flat)  # Ensure probabilities
        p = pred_flat
        p_t = torch.where(target_flat == 1, p, 1-p)  # p_t = p if target=1, else 1-p
        
        # Focal loss formula
        focal_loss = -(1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)  # +eps for numerical stability
        focal_loss = focal_loss.mean()

        return self.alpha * dice_loss + self.beta * focal_loss