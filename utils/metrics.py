import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from skimage.feature import graycomatrix, graycoprops


def evaluate_metrics(original, denoised):
    """Fixed metrics calculation"""
    try:
        # Ensure same dimensions
        if original.shape != denoised.shape:
            denoised = cv2.resize(denoised, (original.shape[1], original.shape[0]))

        # Ensure proper type conversion
        original = torch.from_numpy(original).float().permute(2, 0, 1).unsqueeze(0)
        denoised = torch.from_numpy(denoised).float().permute(2, 0, 1).unsqueeze(0)

        # Calculate metrics
        mse = F.mse_loss(original, denoised).item()
        psnr = -10 * np.log10(mse) if mse > 0 else 100

        ssim_val = ssim(original, denoised, data_range=1.0)

        return {
            'PSNR': psnr,
            'SSIM': ssim_val.item(),
        }

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'PSNR': 0.0,
            'SSIM': 0.0,
        }


def compute_texture_score(original, denoised):
    """Calculate texture preservation score between original and denoised images"""

    def get_glcm_features(img):
        # Ensure proper data type conversion for OpenCV
        if img.dtype != np.uint8:
            # Scale to 0-255 range and convert to uint8
            img = (np.clip(img * 255, 0, 255)).astype(np.uint8)

        # Convert to grayscale if the input is RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 2:
            # If already grayscale, ensure it's uint8
            img = img.astype(np.uint8)

        # Calculate GLCM
        glcm = graycomatrix(img, distances=[1],
                            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=256,
                            symmetric=True,
                            normed=True)

        # Calculate GLCM properties
        features = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in properties:
            features.append(graycoprops(glcm, prop))

        return np.concatenate(features)

    # Get features for both images
    original_features = get_glcm_features(original)
    denoised_features = get_glcm_features(denoised)

    # Calculate similarity score (1 - normalized difference)
    score = 1 - np.mean(np.abs(original_features - denoised_features))
    return score
