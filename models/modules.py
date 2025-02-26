import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """改进的残差块，用于细节处理"""

    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class NonLocalBlock(nn.Module):
    """非局部注意力模块，用于捕获长距离依赖"""

    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.inter_channels = channels // 2

        self.g = nn.Conv2d(channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(channels, self.inter_channels, 1)

        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, channels, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        return x + W_y


class AdvancedResBlock(nn.Module):
    def __init__(self, channels):
        super(AdvancedResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),  # 替换为Instance Normalization
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

        # 增加密集连接
        self.dense1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 2, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.PReLU()
        )

        self.dense2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 3, channels, 3),
            nn.InstanceNorm2d(channels)
        )

        # 改进的注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.PReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(2, 1, 7),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x

        # 密集连接
        out1 = self.conv1(x)
        cat1 = torch.cat([x, out1], dim=1)
        out2 = self.dense1(cat1)
        cat2 = torch.cat([x, out1, out2], dim=1)
        out3 = self.dense2(cat2)

        # 双重注意力
        ca = self.channel_attention(out3)
        out = out3 * ca

        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        out = out * sa

        return out + res


class DetailAwareResBlock(nn.Module):
    def __init__(self, channels):
        super(DetailAwareResBlock, self).__init__()

        # Main convolution path
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

        # Detail-aware path
        self.detail_branch = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels // 2, 3, groups=channels // 2),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

        # Local context aggregation
        self.context_agg = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(channels, channels // 4, 1),
            nn.PReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x

        # Main path
        out = self.conv1(x)
        out = self.conv2(out)

        # Detail-aware processing
        detail_mask = self.detail_branch(x)
        out = out * detail_mask

        # Context modulation
        context = self.context_agg(out)
        out = out * context

        return out + res


class EnhancedDetailPreservationModule(nn.Module):
    def __init__(self, channels):
        super(EnhancedDetailPreservationModule, self).__init__()

        # Multi-scale feature extraction with different receptive fields
        self.branch1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels // 4, 3, groups=channels // 4),
            nn.InstanceNorm2d(channels // 4),
            nn.PReLU()
        )

        self.branch2 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(channels, channels // 4, 5, dilation=2),
            nn.InstanceNorm2d(channels // 4),
            nn.PReLU()
        )

        self.branch3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(channels, channels // 4, 7, dilation=3),
            nn.InstanceNorm2d(channels // 4),
            nn.PReLU()
        )

        self.branch4 = nn.Sequential(
            nn.ReflectionPad2d(6),
            nn.Conv2d(channels, channels // 4, 9, dilation=4),
            nn.InstanceNorm2d(channels // 4),
            nn.PReLU()
        )

        # Advanced edge detection with different scales
        self.edge_detect1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, groups=channels // 2),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels // 2, 1)
        )

        self.edge_detect2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 5, padding=2, groups=channels // 2),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels // 2, 1)
        )

        # Texture preservation module
        self.texture_preserve = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 1)
        )

        # Detail enhancement module
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Sigmoid()
        )

        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.PReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # Final refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        # 只是用于调试的打印
        # print(f"Detail module input shape: {x.shape}")

        # Multi-scale feature extraction
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)

        # Concatenate multi-scale features
        multi_scale = torch.cat([feat1, feat2, feat3, feat4], dim=1)

        # Edge detection at different scales
        edge1 = self.edge_detect1(x)
        edge2 = self.edge_detect2(x)
        edges = torch.cat([edge1, edge2], dim=1)

        # Texture preservation
        texture_features = self.texture_preserve(torch.cat([multi_scale, edges], dim=1))

        # Detail enhancement
        enhanced_details = self.detail_enhance(torch.cat([texture_features, x], dim=1))

        # Channel attention
        channel_att = self.channel_attention(enhanced_details)
        enhanced_details = enhanced_details * channel_att

        # Spatial attention
        avg_out = torch.mean(enhanced_details, dim=1, keepdim=True)
        max_out, _ = torch.max(enhanced_details, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        enhanced_details = enhanced_details * spatial_att

        # Final refinement
        output = self.refinement(torch.cat([enhanced_details, x], dim=1))

        # Residual connection
        return x + output * 0.1  # Scale factor to prevent over-enhancement


class ModelEMA:
    """模型指数移动平均"""

    def __init__(self, model, decay=0.999):
        import copy
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
