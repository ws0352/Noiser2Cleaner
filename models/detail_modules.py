import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ResBlock, NonLocalBlock


class TextureAnalyzer(nn.Module):
    """纹理分析模块，用于指导细节生成"""

    def __init__(self, in_channels):
        super(TextureAnalyzer, self).__init__()

        # 梯度特征提取
        self.gradient_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.PReLU()
            ) for _ in range(4)  # 4个方向的梯度
        ])

        # 纹理特征聚合
        self.texture_aggregator = nn.Sequential(
            nn.Conv2d(in_channels * 4, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取不同方向的梯度特征
        gradient_features = []
        for extractor in self.gradient_extractor:
            gradient = extractor(x)
            gradient_features.append(gradient)

        # 合并梯度特征
        combined_gradients = torch.cat(gradient_features, dim=1)

        # 聚合纹理信息
        texture_guidance = self.texture_aggregator(combined_gradients)
        return texture_guidance


class DetailEnhancer(nn.Module):
    """细节增强模块，用于增强生成的细节"""

    def __init__(self):
        super(DetailEnhancer, self).__init__()

        # 自适应细节增强
        self.enhancement_network = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  # 输入是细节图和纹理指导
            nn.PReLU(),
            ResBlock(32),
            NonLocalBlock(32),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

        # 细节频率调制
        self.frequency_modulator = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.PReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, details, texture_guidance):
        # 合并细节和纹理信息
        combined = torch.cat([details, texture_guidance], dim=1)

        # 增强细节
        enhanced = self.enhancement_network(combined)

        # 调制细节频率
        frequency_weights = self.frequency_modulator(enhanced)
        modulated_details = enhanced * frequency_weights

        return modulated_details


class DetailSynthesisModule(nn.Module):
    """高级细节合成模块，用于生成和增强图像细节"""

    def __init__(self, in_channels):
        super(DetailSynthesisModule, self).__init__()

        # 使用更标准的CNN架构以避免尺寸问题
        self.main_path = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, 32, 3),
            nn.InstanceNorm2d(32),
            nn.PReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3),
            nn.InstanceNorm2d(64),
            nn.PReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3),
            nn.InstanceNorm2d(32),
            nn.PReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, in_channels, 3),
            nn.Tanh()
        )

        # 细节调制器
        self.detail_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 16, 1),
            nn.PReLU(),
            nn.Conv2d(16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 主路径
        details = self.main_path(x)

        # 细节调制
        weights = self.detail_modulator(details)
        modulated_details = details * weights

        # 残差连接 - 添加到原始输入
        return x + modulated_details * 0.1  # 缩放系数以避免过度增强
