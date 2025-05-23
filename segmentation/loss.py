import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        self.eps = 1e-6  # Small epsilon to prevent numerical instability

    def forward(self, inputs, targets):
        # Apply softmax and ensure values are in valid range
        inputs = torch.softmax(inputs, dim=1)
        inputs = torch.clamp(inputs, self.eps, 1.0 - self.eps)

        # Ensure targets are in valid range for BCE
        targets = torch.clamp(targets, 0.0, 1.0)

        # Use binary_cross_entropy with numerical stability
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        p_t = (inputs * targets) + ((1 - inputs) * (1 - targets))
        p_t = torch.clamp(p_t, self.eps, 1.0)  # Prevent pow(0, gamma)

        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Apply reduction with safeguards
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN or Inf detected in loss calculation")
            return torch.tensor(0.1, device=inputs.device, requires_grad=True)

        return loss

class Deep_Supervised_Loss(nn.Module):
    def __init__(self):
        super(Deep_Supervised_Loss, self).__init__()
        self.loss = FocalLoss()

    def forward(self, input, target):
        loss = 0
        # Handle different types of inputs
        if isinstance(input, list):
            for i, img in enumerate(input):
                w = 1 / (2 ** i)
                label = F.interpolate(target, img.size()[2:], mode='nearest')
                l = self.loss(img, label)
                loss += l * w
        else:
            # Single input case
            loss = self.loss(input, target)

        return loss
