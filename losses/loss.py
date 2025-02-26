import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ssim


class RefinedLoss(nn.Module):
    def __init__(self):
        super(RefinedLoss, self).__init__()
        self.vgg = self.build_vgg_features()
        self.criterion = nn.L1Loss()

    def build_vgg_features(self):
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:35]
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg.eval()

    def forward(self, pred, target):
        # Basic reconstruction loss
        l1_loss = self.criterion(pred, target)

        # Perceptual loss using VGG features
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        perceptual_loss = self.criterion(pred_features, target_features)

        # Single-scale SSIM loss instead of MS-SSIM
        ssim_val = ssim(pred, target, data_range=1.0)
        ssim_loss = 1 - ssim_val

        # Gradient loss
        pred_grad = self.compute_gradient(pred)
        target_grad = self.compute_gradient(target)
        gradient_loss = self.criterion(pred_grad, target_grad)

        # Weighted combination
        total_loss = (
                0.35 * l1_loss +
                0.25 * perceptual_loss +
                0.25 * ssim_loss +
                0.15 * gradient_loss
        )

        return total_loss

    def compute_gradient(self, x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3).to(x.device)

        grad_x = F.conv2d(x, sobel_x.expand(x.shape[1], 1, 3, 3), groups=x.shape[1], padding=1)
        grad_y = F.conv2d(x, sobel_y.expand(x.shape[1], 1, 3, 3), groups=x.shape[1], padding=1)

        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
