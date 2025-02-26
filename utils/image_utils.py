import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.amp


def create_gaussian_weight_mask(size, sigma=None):
    """创建高斯权重掩码"""
    if sigma is None:
        sigma = size / 6

    center = size / 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    mask = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return np.expand_dims(mask, axis=2)


def create_detail_mask(img):
    """Create detail area mask with proper data type handling"""
    # Ensure input is uint8
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 255)).astype(np.uint8)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Calculate variance
    mean = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3)
    mean_sq = cv2.GaussianBlur((img.astype(np.float32)) ** 2, (0, 0), 3)
    variance = mean_sq - mean * mean

    # Create mask
    mask = np.exp(-variance / (variance.max() + 1e-6))
    return cv2.GaussianBlur(mask, (0, 0), 3)


def load_and_process_image(image_path):
    """Safe image loading and preprocessing"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None


def denoise_image(model, image_path, device='cuda', tile_size=128, tile_overlap=32):
    """Enhanced image denoising function with improved error handling and size validation"""
    model.eval()

    try:
        # Image preprocessing with enhanced error checking
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Check for zero-sized dimensions
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"Invalid image dimensions: {h}x{w}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Ensure minimum dimensions with safety checks
        min_size = tile_size + 2 * tile_overlap
        if h < min_size or w < min_size:
            scale = float(min_size) / float(min(h, w))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid scale factor: {scale}")

            new_h = max(min_size, int(h * scale))
            new_w = max(min_size, int(w * scale))

            # Additional size validation
            if new_h <= 0 or new_w <= 0:
                raise ValueError(f"Invalid resized dimensions: {new_h}x{new_w}")

            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = image.shape[:2]

        # Calculate padding
        pad_h = (tile_size - (h - 2 * tile_overlap) % tile_size) % tile_size
        pad_w = (tile_size - (w - 2 * tile_overlap) % tile_size) % tile_size

        # Add padding with size validation
        padded_h = h + 2 * tile_overlap + pad_h
        padded_w = w + 2 * tile_overlap + pad_w

        if padded_h <= 0 or padded_w <= 0:
            raise ValueError(f"Invalid padded dimensions: {padded_h}x{padded_w}")

        padded_image = np.pad(image,
                              ((tile_overlap, tile_overlap + pad_h),
                               (tile_overlap, tile_overlap + pad_w),
                               (0, 0)),
                              mode='reflect')

        # Initialize result arrays
        denoised = np.zeros_like(padded_image)
        weight_map = np.zeros(padded_image.shape[:2] + (1,))

        # Create weight mask
        weight_mask = create_gaussian_weight_mask(tile_size)

        # Process tiles with enhanced error checking
        with torch.no_grad():
            y_steps = range(0, padded_image.shape[0] - tile_size + 1, tile_size - 2 * tile_overlap)
            x_steps = range(0, padded_image.shape[1] - tile_size + 1, tile_size - 2 * tile_overlap)

            if not y_steps or not x_steps:
                raise ValueError("No valid steps for tile processing")

            for y in y_steps:
                for x in x_steps:
                    # Extract and validate tile
                    tile = padded_image[y:y + tile_size, x:x + tile_size]
                    if tile.shape[:2] != (tile_size, tile_size):
                        print(f"Warning: Skipping invalid tile at ({x}, {y}): {tile.shape}")
                        continue

                    # Process tile with error handling
                    try:
                        tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(device)

                        with torch.amp.autocast(device_type='cuda', enabled=(device == 'cuda')):
                            denoised_tile = model(tile_tensor)

                        denoised_tile = denoised_tile.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        denoised[y:y + tile_size, x:x + tile_size] += denoised_tile * weight_mask
                        weight_map[y:y + tile_size, x:x + tile_size] += weight_mask
                    except Exception as e:
                        print(f"Error processing tile at ({x}, {y}): {str(e)}")
                        continue

        # Check if any tiles were processed
        if np.all(weight_map == 0):
            raise ValueError("No tiles were successfully processed")

        # Normalize and crop result with safety checks
        epsilon = 1e-8
        if np.any(weight_map < epsilon):
            print("Warning: Some areas have very low weights")
        denoised = denoised / (weight_map + epsilon)

        # Validate final crop dimensions
        if tile_overlap + h > denoised.shape[0] or tile_overlap + w > denoised.shape[1]:
            raise ValueError("Invalid crop dimensions")

        denoised = denoised[tile_overlap:tile_overlap + h, tile_overlap:tile_overlap + w]

        # Post-process with error handling
        try:
            enhanced = post_process(denoised)
        except Exception as e:
            print(f"Warning: Post-processing failed: {str(e)}")
            enhanced = denoised

        return np.clip(enhanced, 0, 1)

    except Exception as e:
        print(f"Error during denoising: {str(e)}")
        raise


def post_process(img):
    """Fixed post-processing function"""
    try:
        # Ensure input is in float32 range 0-1
        img = np.clip(img, 0, 1).astype(np.float32)

        # Convert to uint8 for OpenCV operations
        img_uint8 = (img * 255).astype(np.uint8)

        # Color space conversion
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Skip guided filter if image is too small
        if l.shape[0] >= 16 and l.shape[1] >= 16:
            try:
                l_enhanced = cv2.GaussianBlur(l, (3, 3), 0)
            except:
                l_enhanced = l
        else:
            l_enhanced = l

        # Merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # Convert back to float32 range 0-1
        final = rgb_enhanced.astype(np.float32) / 255.0
        return np.clip(final, 0, 1)

    except Exception as e:
        print(f"Error in post-processing: {str(e)}")
        return img
