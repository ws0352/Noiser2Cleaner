import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

img = cv2.imread('OIP.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))
img = img / 255.0



class DABlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(DABlock, self).__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_attn = self.spatial_attn(x)
        channel_attn = self.channel_attn(x)
        out = spatial_attn * channel_attn * x
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class DATransUNet(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DATransUNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(2048)
        # self.da1 = DABlock(64, 64)
        # self.da2 = DABlock(128, 128)
        # self.da3 = DABlock(256, 256)
        # self.da4 = DABlock(512, 512)
        # self.da5 = DABlock(1024, 1024)
        self.da1 = DABlock(128)
        self.da2 = DABlock(256)
        self.da3 = DABlock(512)
        self.da4 = DABlock(1024)
        self.da5 = DABlock(2048)
        self.trans_encoder = nn.TransformerEncoder(TransformerEncoderLayer(d_model=2048, nhead=nhead), num_layers=num_encoder_layers)
        self.trans_decoder = nn.TransformerDecoder(TransformerDecoderLayer(d_model=2048, nhead=nhead), num_layers=num_decoder_layers)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(1024)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv10 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv12 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = self.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = self.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = self.relu(self.bn5(self.conv5(x5)))
        x1 = self.da1(x1)
        x2 = self.da2(x2)
        x3 = self.da3(x3)
        x4 = self.da4(x4)
        x5 = self.da5(x5)
        b, c, h, w = x5.size()
        features = []
        x = x5.view(b, c, -1).permute(2, 0, 1)
        x = self.trans_encoder(x)
        x = self.trans_decoder(x, x, memory_mask=None)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        x = self.up1(x)
        x = self.relu(self.bn6(self.conv6(x)))
        features.append(x)
        x = torch.cat([x4, x], dim=1)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.up2(x)
        x = self.relu(self.bn8(self.conv8(x)))
        features.append(x)
        x = torch.cat([x3, x], dim=1)
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.up3(x)
        x = self.relu(self.bn10(self.conv10(x)))
        features.append(x)
        x = torch.cat([x2, x], dim=1)
        x = self.relu(self.bn11(self.conv11(x)))
        x = self.up4(x)
        x = self.relu(self.bn12(self.conv12(x)))
        features.append(x)
        x = torch.cat([x1, x], dim=1)
        x = self.relu(self.bn13(self.conv13(x)))
        x = self.conv14(x)
        return x, features




def compute_spectrum(image):
    # Convert image data to grayscale
    if image.shape[2] == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    # Compute the 2D Fourier transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Add 1 to avoid log of zero
    return magnitude_spectrum

def plot_spectrum_comparison(spectrum1, spectrum2, title1='Input Image Spectrum', title2='Output Image Spectrum'):
    # Set the font to Times New Roman
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    
    plt.figure(figsize=(8, 8))
    plt.plot(spectrum1[int(spectrum1.shape[0]/2)], label=title1, color='blue', linewidth=2)
    plt.plot(spectrum2[int(spectrum2.shape[0]/2)], label=title2, color='red', linewidth=2)
    plt.title('Spectrum Comparison', fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()



import os

def plot_all_spectra(spectra, titles):
    # Set the font to Times New Roman
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    
    plt.figure(figsize=(10, 8))
    for spectrum, title in zip(spectra, titles):
        plt.plot(spectrum[int(spectrum.shape[0]/2)], label=title, linewidth=2)
    plt.title('Spectrum Comparison of All Models', fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def process_and_plot_spectra_for_all_models(directory, image):
    spectra = []
    titles = []
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().cuda()
    spectra.append(compute_spectrum(image))
    titles.append(f'Spectrum of Input Image')
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            model_path = os.path.join(directory, filename)
            model = torch.load(model_path).cuda()
            output_img, _ = model(x)
            output_img = output_img[0].detach().cpu().permute(1, 2, 0).numpy()
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            output_img = np.clip(output_img * 255, 0, 255)
            output_spectrum = compute_spectrum(output_img)
            spectra.append(output_spectrum)
            titles.append(f'Spectrum of {filename[:-4]}')
    
    plot_all_spectra(spectra, titles)

# Assuming 'img' is already loaded and preprocessed as per the earlier code
process_and_plot_spectra_for_all_models('./', img)