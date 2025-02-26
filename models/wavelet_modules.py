import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletDecomposition(nn.Module):
    """Enhanced wavelet decomposition with fixed dimensions"""

    def __init__(self):
        super(WaveletDecomposition, self).__init__()
        # Initialize filters with correct dimensions
        self.lp_filter = nn.Parameter(self._initialize_lowpass_filter())
        self.hp_filter = nn.Parameter(self._initialize_highpass_filter())

        # Coefficient enhancement layers
        self.coeff_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.PReLU(),
                nn.Conv2d(16, 1, 3, padding=1)
            ) for _ in range(4)  # One for each subband
        ])

        # Adaptive thresholding parameters
        self.threshold = nn.Parameter(torch.tensor([0.1]))
        self.soft_factor = nn.Parameter(torch.tensor([2.0]))

    def _initialize_lowpass_filter(self):
        """Initialize lowpass filter with correct shape"""
        filter_1d = torch.tensor([
            0.0033, -0.0126, -0.0062, 0.0776, 0.0322, -0.2423,
            -0.1384, 0.7243, 0.7243, -0.1384, -0.2423, 0.0322,
            0.0776, -0.0062, -0.0126, 0.0033
        ])
        # Reshape to 2D filter: [out_channels, in_channels/groups, height, width]
        return filter_1d.reshape(1, 1, -1, 1)

    def _initialize_highpass_filter(self):
        """Initialize highpass filter with correct shape"""
        filter_1d = torch.tensor([
            -0.0033, 0.0126, -0.0062, -0.0776, 0.0322, 0.2423,
            -0.1384, -0.7243, 0.7243, 0.1384, -0.2423, -0.0322,
            0.0776, 0.0062, -0.0126, -0.0033
        ])
        return filter_1d.reshape(1, 1, -1, 1)

    def _adaptive_threshold(self, x):
        """Apply adaptive soft thresholding"""
        magnitude = torch.abs(x)
        threshold = self.threshold * torch.std(x)
        scale = F.softplus(self.soft_factor)
        soft_thresh = torch.sign(x) * F.relu(magnitude - threshold) * scale
        return soft_thresh

    def _apply_separable_filter(self, x, h_filter, v_filter=None):
        """Apply 2D separable wavelet transform with correct dimensions"""
        batch_size, channels, height, width = x.size()
        if v_filter is None:
            v_filter = h_filter

        # Pad signal
        pad_size = (h_filter.size(2) - 1) // 2
        x_pad = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

        # 确保尺寸是偶数，这对于下采样很重要
        _, _, pad_h, pad_w = x_pad.size()
        if pad_h % 2 == 1:
            x_pad = F.pad(x_pad, (0, 0, 0, 1), mode='reflect')
        if pad_w % 2 == 1:
            x_pad = F.pad(x_pad, (0, 1, 0, 0), mode='reflect')

        # Apply horizontal filter
        x_h = F.conv2d(
            x_pad,
            h_filter.repeat(channels, 1, 1, 1),
            stride=(1, 2),
            groups=channels
        )

        # Apply vertical filter
        x_hv = F.conv2d(
            x_h,
            v_filter.transpose(2, 3).repeat(channels, 1, 1, 1),
            stride=(2, 1),
            groups=channels
        )

        return x_hv

    def forward(self, x):
        # Apply separable wavelet transform
        LL = self._apply_separable_filter(x, self.lp_filter, self.lp_filter)
        LH = self._apply_separable_filter(x, self.lp_filter, self.hp_filter)
        HL = self._apply_separable_filter(x, self.hp_filter, self.lp_filter)
        HH = self._apply_separable_filter(x, self.hp_filter, self.hp_filter)

        # Enhance coefficients
        for i, subb in enumerate([LL, LH, HL, HH]):
            # Process each channel separately
            enhanced = []
            for c in range(subb.size(1)):
                band = subb[:, c:c + 1]
                enhanced.append(self.coeff_enhance[i](band))
            subb = torch.cat(enhanced, dim=1)

            # Apply threshold to detail coefficients
            if i > 0:  # Skip LL band
                subb = self._adaptive_threshold(subb)

            if i == 0:
                LL = subb
            elif i == 1:
                LH = subb
            elif i == 2:
                HL = subb
            else:
                HH = subb

        return LL / 4.0, LH / 4.0, HL / 4.0, HH / 4.0


class WaveletReconstruction(nn.Module):
    """Enhanced wavelet reconstruction with correct dimensions"""

    def __init__(self):
        super(WaveletReconstruction, self).__init__()

        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

        # Channel-wise coefficient processing
        self.detail_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.PReLU(),
                nn.Conv2d(16, 1, 3, padding=1, bias=False)
            ) for _ in range(3)
        ])

        # Edge preservation module
        self.edge_preserve = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

        # Final enhancement
        self.final_enhance = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )

    def _enhance_coefficients(self, x, module_idx):
        """Process coefficients channel-wise"""
        enhanced = []
        for c in range(x.size(1)):
            band = x[:, c:c + 1]
            enhanced.append(self.detail_enhance[module_idx](band))
        return torch.cat(enhanced, dim=1)

    def forward(self, LL, LH, HL, HH):
        # Process detail coefficients
        LH_enhanced = self._enhance_coefficients(LH, 0)
        HL_enhanced = self._enhance_coefficients(HL, 1)
        HH_enhanced = self._enhance_coefficients(HH, 2)

        # Apply adaptive coefficient mixing
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        gamma = torch.sigmoid(self.gamma)

        # Edge mask generation
        edge_mask = self.edge_preserve(torch.cat([
            LH_enhanced.mean(dim=1, keepdim=True),
            HL_enhanced.mean(dim=1, keepdim=True),
            HH_enhanced.mean(dim=1, keepdim=True)
        ], dim=1))

        # Combine coefficients
        LH_final = LH + beta * LH_enhanced * edge_mask[:, 0:1]
        HL_final = HL + beta * HL_enhanced * edge_mask[:, 1:2]
        HH_final = HH + beta * HH_enhanced * edge_mask[:, 2:3]

        # Inverse transform with size matching
        L = torch.cat([LL + alpha * LH_final, LL - alpha * LH_final], dim=2)
        H = torch.cat([HL_final + gamma * HH_final, HL_final - gamma * HH_final], dim=2)
        x = torch.cat([L + H, L - H], dim=3)

        # Apply final enhancement
        return self.final_enhance(x)


class EdgePreservationModule(nn.Module):
    """Module for preserving edges during reconstruction"""

    def __init__(self):
        super(EdgePreservationModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 16, 3, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 4, 1),
            nn.PReLU(),
            nn.Conv2d(4, 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv1(x)
        att = self.attention(feat)
        feat = feat * att
        return self.conv2(feat)
