from torch import nn

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Binary cross-entropy
        bce_loss = self.bce(inputs, targets)

        # Dice loss
        intersection = (inputs * targets).sum(dim=0)
        dice_loss = 1 - ((2. * intersection + self.smooth) /
                         (inputs.sum(dim=0) + targets.sum(dim=0) + self.smooth)).mean()

        return bce_loss + dice_loss, 1-dice_loss
