import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImprovedDataset(Dataset):
    def __init__(self, image_path, patch_size=192, num_samples=2000, aug_prob=0.7):
        super(ImprovedDataset, self).__init__()
        # Read image and convert to RGB with proper dtype
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # Convert to float32 and normalize to [0, 1]
        self.image = self.image.astype(np.float32) / 255.0

        self.patch_size = patch_size
        self.num_samples = num_samples
        self.aug_prob = aug_prob

        # Ensure image is large enough for patch extraction
        if self.image.shape[0] < patch_size or self.image.shape[1] < patch_size:
            scale_factor = max(patch_size / self.image.shape[0], patch_size / self.image.shape[1]) * 1.1
            new_size = (int(self.image.shape[1] * scale_factor), int(self.image.shape[0] * scale_factor))
            self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_CUBIC)

        # Extract patches
        self.patches = self.extract_patches()

    def calculate_gradient_map(self):
        """Calculate gradient map with proper dtype handling"""
        # Convert to uint8 for OpenCV operations
        img_uint8 = (self.image * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        return cv2.GaussianBlur(grad_mag, (7, 7), 0)

    def extract_patches(self):
        h, w = self.image.shape[:2]
        patches = []

        # Calculate gradient map for intelligent sampling
        grad_map = self.calculate_gradient_map()
        grad_map = grad_map / (grad_map.sum() + 1e-8)  # Normalize with epsilon

        for _ in range(self.num_samples):
            if np.random.rand() < 0.7:  # 70% chance to select high gradient area
                # Flatten for random choice
                flat_idx = np.random.choice(grad_map.size, p=grad_map.flatten())
                y, x = np.unravel_index(flat_idx, grad_map.shape)

                # Adjust coordinates to ensure valid patch extraction
                y = np.clip(y - self.patch_size // 2, 0, h - self.patch_size)
                x = np.clip(x - self.patch_size // 2, 0, w - self.patch_size)
            else:
                y = np.random.randint(0, h - self.patch_size)
                x = np.random.randint(0, w - self.patch_size)

            patch = self.image[y:y + self.patch_size, x:x + self.patch_size].copy()
            patches.append(patch)

        return patches

    def advanced_augmentation(self, img):
        """Apply augmentations with proper dtype handling"""
        # Ensure float32 type
        img = img.astype(np.float32)

        # Geometric transformations
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
        if np.random.rand() < 0.5:
            img = np.flipud(img)
        if np.random.rand() < 0.5:
            k = np.random.randint(4)
            img = np.rot90(img, k)

        # Color augmentation
        if np.random.rand() < 0.3:
            # Convert to uint8 for color space conversion
            img_uint8 = (img * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

            # Apply adjustments in HSV space
            hsv = hsv.astype(np.float32)
            hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.rand() * 0.4)  # Hue
            hsv[:, :, 1] = hsv[:, :, 1] * (0.8 + np.random.rand() * 0.4)  # Saturation
            hsv[:, :, 2] = hsv[:, :, 2] * (0.8 + np.random.rand() * 0.4)  # Value

            # Convert back to uint8 for color space conversion
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        return np.clip(img, 0, 1)

    def add_realistic_noise(self, img):
        """Add realistic noise patterns with proper data type and range handling"""
        # Ensure float32 type and proper range
        img = np.clip(img, 0, 1).astype(np.float32)

        # Convert to uint8 for color space conversion
        img_uint8 = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # Calculate local variance with proper normalization
        gray_float = gray.astype(np.float32) / 255.0
        local_var = cv2.GaussianBlur(gray_float ** 2, (7, 7), 0) - cv2.GaussianBlur(gray_float, (7, 7), 0) ** 2
        local_var = np.clip(local_var, 0, None)  # Ensure non-negative variance

        # Normalize variance for noise scaling
        noise_level = np.exp(-local_var * 10) * 0.1
        noise_level = np.clip(noise_level, 0.01, 0.1)  # Limit noise range

        noisy = img.copy()

        # Add Gaussian noise with proper scaling
        gaussian_noise = np.random.normal(0, 0.05, img.shape) * np.expand_dims(noise_level, -1)
        noisy = np.clip(noisy + gaussian_noise, 0, 1)

        # Add Poisson noise with proper scaling
        # Scale image to appropriate range for Poisson noise
        scaled = (noisy * 255.0).clip(0, 255)
        poisson_noise = (np.random.poisson(scaled + 1) - (scaled + 1)) / 255.0
        noisy = np.clip(noisy + poisson_noise * 0.1, 0, 1)

        # Add salt and pepper noise with controlled density
        salt_pepper_prob = 0.01 * np.clip(1 - local_var / (local_var.max() + 1e-10), 0, 1)
        # Create mask with proper broadcasting
        salt_pepper_mask = np.random.random(img.shape) < np.expand_dims(salt_pepper_prob, -1)
        # Apply salt and pepper values directly
        salt_pepper_values = np.random.choice([0, 1], size=img.shape)
        noisy = np.where(salt_pepper_mask, salt_pepper_values, noisy)

        # Optional JPEG compression simulation
        if np.random.rand() < 0.5:
            img_uint8 = (np.clip(noisy * 255, 0, 255)).astype(np.uint8)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(60, 95)]
            _, encimg = cv2.imencode('.jpg', img_uint8, encode_param)
            noisy = cv2.imdecode(encimg, 1).astype(np.float32) / 255.0

        return np.clip(noisy, 0, 1)

    def __getitem__(self, idx):
        clean_patch = self.patches[idx].copy()

        # Apply augmentation
        if np.random.rand() < self.aug_prob:
            clean_patch = self.advanced_augmentation(clean_patch)

        # Add noise
        noisy_patch = self.add_realistic_noise(clean_patch)

        # Ensure float32 type before converting to tensor
        return (
            torch.from_numpy(noisy_patch.astype(np.float32)).permute(2, 0, 1).float(),
            torch.from_numpy(clean_patch.astype(np.float32)).permute(2, 0, 1).float()
        )

    def __len__(self):
        return len(self.patches)
