from torch import nn
import torch

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):

        targets_binary = torch.argmax(targets, dim=1, keepdim=True).squeeze()
        inputs_binary  = torch.argmax(inputs, dim=1, keepdim=True).squeeze()

        # Dice loss
        intersection = (inputs_binary * targets_binary)
        dice_loss = 1 - ((2 * intersection) /
                         (inputs_binary + targets_binary + self.smooth)).mean()
        
        return dice_loss, 1- dice_loss
