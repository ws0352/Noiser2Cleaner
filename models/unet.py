import torch
import torch.nn as nn
import torch.nn.functional as F
from .detail_modules import DetailSynthesisModule
from .modules import EnhancedDetailPreservationModule


class SimpleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SimpleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.InstanceNorm2d(out_ch),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ImprovedUNet, self).__init__()

        # 简化模型 - 使用固定尺寸的卷积块
        # Encoder path
        self.enc1 = SimpleConvBlock(in_channels, 64)
        self.pool1 = nn.AvgPool2d(2)

        self.enc2 = SimpleConvBlock(64, 128)
        self.pool2 = nn.AvgPool2d(2)

        self.enc3 = SimpleConvBlock(128, 256)
        self.pool3 = nn.AvgPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            SimpleConvBlock(256, 512),
            SimpleConvBlock(512, 256)
        )

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = SimpleConvBlock(512, 128)  # 256 + 256 concat

        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = SimpleConvBlock(256, 64)  # 128 + 128 concat

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = SimpleConvBlock(128, 64)  # 64 + 64 concat

        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Detail synthesis (如果需要的话)
        self.detail_generator = DetailSynthesisModule(in_channels=3)

    def forward(self, x):
        # 确保输入尺寸是8的倍数以防止尺寸不匹配
        _, _, h, w = x.size()
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 记录每个步骤的尺寸用于调试
        # print(f"Padded input shape: {x.shape}")

        # Encoder
        e1 = self.enc1(x)
        # print(f"e1 shape: {e1.shape}")

        e2 = self.enc2(self.pool1(e1))
        # print(f"e2 shape: {e2.shape}")

        e3 = self.enc3(self.pool2(e2))
        # print(f"e3 shape: {e3.shape}")

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        # print(f"bottleneck shape: {b.shape}")

        # Decoder with precise size control
        d3_up = self.upconv3(b)
        # print(f"d3_up shape: {d3_up.shape}, e3 shape: {e3.shape}")
        # 确保上采样后的尺寸与跳跃连接匹配
        if d3_up.shape[2:] != e3.shape[2:]:
            d3_up = F.interpolate(d3_up, size=e3.shape[2:], mode='bilinear', align_corners=True)
            # print(f"d3_up adjusted shape: {d3_up.shape}")

        d3 = self.dec3(torch.cat([d3_up, e3], dim=1))
        # print(f"d3 shape: {d3.shape}")

        d2_up = self.upconv2(d3)
        # print(f"d2_up shape: {d2_up.shape}, e2 shape: {e2.shape}")
        # 确保上采样后的尺寸与跳跃连接匹配
        if d2_up.shape[2:] != e2.shape[2:]:
            d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode='bilinear', align_corners=True)
            # print(f"d2_up adjusted shape: {d2_up.shape}")

        d2 = self.dec2(torch.cat([d2_up, e2], dim=1))
        # print(f"d2 shape: {d2.shape}")

        d1_up = self.upconv1(d2)
        # print(f"d1_up shape: {d1_up.shape}, e1 shape: {e1.shape}")
        # 确保上采样后的尺寸与跳跃连接匹配
        if d1_up.shape[2:] != e1.shape[2:]:
            d1_up = F.interpolate(d1_up, size=e1.shape[2:], mode='bilinear', align_corners=True)
            # print(f"d1_up adjusted shape: {d1_up.shape}")

        d1 = self.dec1(torch.cat([d1_up, e1], dim=1))
        # print(f"d1 shape: {d1.shape}")

        # Final output
        out = self.final_conv(d1)
        # print(f"output shape: {out.shape}")

        # 去除填充，确保输出尺寸与输入相同
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
            # print(f"final output shape (after crop): {out.shape}")

        # Detail enhancement
        enhanced_out = self.detail_generator(out)
        # print(f"enhanced output shape: {enhanced_out.shape}")

        return enhanced_out