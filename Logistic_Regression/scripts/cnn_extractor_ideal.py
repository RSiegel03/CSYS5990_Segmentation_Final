import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        # First convolutional layer with 8 filters
        self.first_conv = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
        self._initialize_first_layer()

        # Sequential max pooling
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 4x4 (for 9x9 input)
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 2x2
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 1x1
        )

    def _initialize_first_layer(self):
        filters = []

        # 1. Sobel - Horizontal
        filters.append(torch.tensor([[[-1, -2, -1],
                                      [ 0,  0,  0],
                                      [ 1,  2,  1]]], dtype=torch.float32))

        # 2. Sobel - Vertical
        filters.append(torch.tensor([[[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]]], dtype=torch.float32))

        # 3. Laplacian
        filters.append(torch.tensor([[[0,  1, 0],
                                      [1, -4, 1],
                                      [0,  1, 0]]], dtype=torch.float32))

        # 4. Emboss
        filters.append(torch.tensor([[[-2, -1, 0],
                                      [-1,  1, 1],
                                      [ 0,  1, 2]]], dtype=torch.float32))

        # 5. Sharpen
        filters.append(torch.tensor([[[ 0, -1,  0],
                                      [-1,  5, -1],
                                      [ 0, -1,  0]]], dtype=torch.float32))

        # 6. Diagonal edge
        filters.append(torch.tensor([[[ 2, -1, -1],
                                      [-1,  2, -1],
                                      [-1, -1,  2]]], dtype=torch.float32))

        # 7. Center-surround
        filters.append(torch.tensor([[[-1, -1, -1],
                                      [-1,  8, -1],
                                      [-1, -1, -1]]], dtype=torch.float32))
        
        #8. General Pixel Strength
        filters.append(torch.tensor([[[ 1,  1,  1],
                                      [ 1,  1,  1],
                                      [ 1,  1,  1]]], dtype=torch.float32))

        weight_tensor = torch.stack(filters)  # shape (8, 1, 3, 3)
        self.first_conv.weight.data = weight_tensor

    def forward(self, x):
        x = self.first_conv(x)  # (B, 8, 9, 9) 
        x = self.features(x)    # (B, 8, 1, 1) after max pooling
        x = x.view(x.size(0), -1)  # Flatten (B, 8)
        return x
