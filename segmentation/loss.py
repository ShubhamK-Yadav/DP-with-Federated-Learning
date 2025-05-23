import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    AMP-safe focal loss for binary or multi-label segmentation.
    Assumes raw logits as input (no softmax or sigmoid applied).
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6  # For numerical stability

    def forward(self, inputs, targets):
        # inputs: raw logits of shape [N, C, H, W]
        # targets: one-hot or binary masks of same shape

        inputs = inputs.clamp(min=-20, max=20)
        # Apply sigmoid to get probabilities
        probas = torch.sigmoid(inputs)

        # Compute BCE with logits directly for stability
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Focal modulation
        p_t = (probas * targets) + ((1 - probas) * (1 - targets))
        focal_weight = (1 - p_t).clamp(min=self.eps) ** self.gamma
        loss = focal_weight * ce_loss

        # Alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class Deep_Supervised_Loss(nn.Module):
    """
    Deep supervision wrapper around FocalLoss.
    Accepts either a single output or a list of outputs for deep supervision.
    """

    def __init__(self):
        super(Deep_Supervised_Loss, self).__init__()
        self.loss = FocalLoss()

    def forward(self, input, target):
        total_loss = 0

        # Deep supervision case
        if isinstance(input, list):
            for i, out in enumerate(input):
                weight = 1 / (2 ** i)
                target_resized = F.interpolate(target, size=out.shape[2:], mode="nearest")
                l = self.loss(out, target_resized)
                total_loss += weight * l
        else:
            total_loss = self.loss(input, target)

        return total_loss
