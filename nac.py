
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models import resnet18
from tqdm.autonotebook import tqdm
from copy import deepcopy

from cka import CKACalculator

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

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv15 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x1 = x # Save for skip connection
        x = self.pool1(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x2 = x # Save for skip connection
        x = self.pool2(x)
        x = self.conv5(x)
        x = nn.ReLU()(x)
        x = self.conv6(x)
        x = nn.ReLU()(x)
        x3 = x # Save for skip connection
        x = self.pool3(x)
        x = self.conv7(x)
        x = nn.ReLU()(x)
        x = self.conv8(x)
        x = nn.ReLU()(x)
        # Decoder
        x = self.upconv1(x)
        x = torch.cat([x, x3], dim=1) # Concatenate with skip connection
        x = self.conv9(x)
        x = nn.ReLU()(x)
        x = self.conv10(x)
        x = nn.ReLU()(x)
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1) # Concatenate with skip connection
        x = self.conv11(x)
        x = nn.ReLU()(x)
        x = self.conv12(x)
        x = nn.ReLU()(x)
        x = self.upconv3(x)
        x = torch.cat([x, x1], dim=1) # Concatenate with skip connection
        x = self.conv13(x)
        x = nn.ReLU()(x)
        x = self.conv14(x)
        x = nn.ReLU()(x)
        x = self.conv15(x)
        x = nn.Sigmoid()(x) # Output range is [0, 1]
        return x
    

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

class DATransUNet1(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=8, num_encoder_layers=6, num_decoder_layers=6, init_channel=32):
        super(DATransUNet1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, init_channel*2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(init_channel*2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(init_channel*2, init_channel*4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(init_channel*4)
        self.conv3 = nn.Conv2d(init_channel*4, init_channel*8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(init_channel*8)
        self.conv4 = nn.Conv2d(init_channel*8, init_channel*16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(init_channel*16)
        self.conv5 = nn.Conv2d(init_channel*16, init_channel*32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(init_channel*32)
        self.da1 = DABlock(init_channel*2)
        self.da2 = DABlock(init_channel*4)
        self.da3 = DABlock(init_channel*8)
        self.da4 = DABlock(init_channel*16)
        self.da5 = DABlock(init_channel*32)
        self.trans_encoder = nn.TransformerEncoder(TransformerEncoderLayer(d_model=init_channel*32, nhead=nhead), num_layers=num_encoder_layers)
        self.trans_decoder = nn.TransformerDecoder(TransformerDecoderLayer(d_model=init_channel*32, nhead=nhead), num_layers=num_decoder_layers)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Conv2d(init_channel*32, init_channel*16, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(init_channel*16)
        self.conv7 = nn.Conv2d(init_channel*32, init_channel*16, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(init_channel*16)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8 = nn.Conv2d(init_channel*16, init_channel*8, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(init_channel*8)
        self.conv9 = nn.Conv2d(init_channel*16, init_channel*8, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(init_channel*8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv10 = nn.Conv2d(init_channel*8, init_channel*4, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(init_channel*4)
        self.conv11 = nn.Conv2d(init_channel*8, init_channel*4, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(init_channel*4)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv12 = nn.Conv2d(init_channel*4, init_channel*2, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(init_channel*2)
        self.conv13 = nn.Conv2d(init_channel*4, init_channel*2, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(init_channel*2)
        self.conv14 = nn.Conv2d(init_channel*2, out_channels, kernel_size=1)

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

class DATransUNet2(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=8, num_encoder_layers=6, num_decoder_layers=6, init_channel=32):
        super(DATransUNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, init_channel*2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(init_channel*2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(init_channel*2, init_channel*4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(init_channel*4)
        self.conv3 = nn.Conv2d(init_channel*4, init_channel*8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(init_channel*8)
        self.conv4 = nn.Conv2d(init_channel*8, init_channel*16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(init_channel*16)
        self.conv5 = nn.Conv2d(init_channel*16, init_channel*32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(init_channel*32)
        self.da1 = DABlock(init_channel*2)
        self.da2 = DABlock(init_channel*4)
        self.da3 = DABlock(init_channel*8)
        self.da4 = DABlock(init_channel*16)
        self.da5 = DABlock(init_channel*32)
        self.trans_encoder = nn.TransformerEncoder(TransformerEncoderLayer(d_model=init_channel*32, nhead=nhead), num_layers=num_encoder_layers)
        self.trans_decoder = nn.TransformerDecoder(TransformerDecoderLayer(d_model=init_channel*32, nhead=nhead), num_layers=num_decoder_layers)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Conv2d(init_channel*32, init_channel*16, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(init_channel*16)
        self.conv7 = nn.Conv2d(init_channel*32, init_channel*16, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(init_channel*16)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8 = nn.Conv2d(init_channel*16, init_channel*8, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(init_channel*8)
        self.conv9 = nn.Conv2d(init_channel*16, init_channel*8, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(init_channel*8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv10 = nn.Conv2d(init_channel*8, init_channel*4, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(init_channel*4)
        self.conv11 = nn.Conv2d(init_channel*8, init_channel*4, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(init_channel*4)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv12 = nn.Conv2d(init_channel*4, init_channel*2, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(init_channel*2)
        self.conv13 = nn.Conv2d(init_channel*4, init_channel*2, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(init_channel*2)
        self.conv14 = nn.Conv2d(init_channel*2, out_channels, kernel_size=1)

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
        # x = self.trans_decoder(x, x, memory_mask=None)
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


if __name__ == '__main__':
    
    model1 = DATransUNet(in_channels=3, out_channels=3).cuda()
    # model = torch.load('model4.pth')
    model1.eval()
    # model2 = DATransUNet1(in_channels=3, out_channels=3).cuda()
    # model2 = DATransUNet2(in_channels=3, out_channels=3).cuda()
    
    model2 = resnet18(pretrained=True).cuda()
    model2.eval()
    # model2 = DATransUNet1(in_channels=3, out_channels=3).cuda()
    transforms = Compose([ToTensor(), 
                          
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = CIFAR10(root='./', train=False, download=True, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    calculator = CKACalculator(model1, model2, dataloader)
    cka_matrix = calculator.calculate_cka_matrix()

    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (7, 7)
    plt.imshow(cka_matrix.cpu().numpy(), cmap='inferno')
    plt.savefig("test4.jpg")
    plt.show()