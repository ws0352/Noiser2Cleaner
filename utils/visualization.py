import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def visualize_results(original, noisy, denoised, save_path=None):
    """Fixed visualization function with proper size handling"""
    try:
        # Ensure all images have the same size by resizing to original dimensions
        target_size = (original.shape[1], original.shape[0])

        if noisy.shape[:2] != original.shape[:2]:
            noisy = cv2.resize(noisy, target_size)
        if denoised.shape[:2] != original.shape[:2]:
            denoised = cv2.resize(denoised, target_size)

        # Ensure proper value range [0, 1]
        original = np.clip(original, 0, 1)
        noisy = np.clip(noisy, 0, 1)
        denoised = np.clip(denoised, 0, 1)

        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(131)
        plt.imshow(original)
        plt.title('Original Image')
        plt.axis('off')

        # Noisy image
        plt.subplot(132)
        plt.imshow(noisy)
        plt.title('Noisy Image')
        plt.axis('off')

        # Denoised image
        plt.subplot(133)
        plt.imshow(denoised)
        plt.title('Denoised Image')
        plt.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        plt.close()


def visualize_training_history(history):
    """Visualize training history with metrics"""
    try:
        plt.figure(figsize=(15, 10))

        # Plot loss
        plt.subplot(221)
        plt.plot(history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)

        # Plot PSNR
        plt.subplot(222)
        plt.plot(history['psnr'])
        plt.title('PSNR')
        plt.xlabel('Epochs')
        plt.ylabel('dB')
        plt.grid(True)

        # Plot SSIM
        plt.subplot(223)
        plt.plot(history['ssim'])
        plt.title('SSIM')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")
        plt.close()


def visualize_comparative_detail(original, noisy, denoised, region=None, save_path=None):
    """Visualize and compare details of original, noisy and denoised images"""
    try:
        # Clone images to prevent modifications
        img_orig = original.copy()
        img_noisy = noisy.copy()
        img_denoised = denoised.copy()

        if region is None:
            # Auto select a region with details (top 25% of high frequency content)
            gray = cv2.cvtColor((img_orig * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Find region with high gradient
            h, w = grad_mag.shape
            region_size = min(h, w) // 4

            # Apply max filter to find area with highest average gradient
            from scipy.ndimage import maximum_filter
            max_filter = maximum_filter(grad_mag, size=region_size)
            y, x = np.unravel_index(np.argmax(max_filter), grad_mag.shape)

            # Ensure region is within bounds
            x = min(max(x - region_size // 2, 0), w - region_size)
            y = min(max(y - region_size // 2, 0), h - region_size)

            region = (x, y, region_size, region_size)

        x, y, w, h = region

        # Extract regions
        region_orig = img_orig[y:y + h, x:x + w]
        region_noisy = img_noisy[y:y + h, x:x + w]
        region_denoised = img_denoised[y:y + h, x:x + w]

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])

        # Plot full images in first row
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(img_orig)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Draw rectangle on region
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)

        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(img_noisy)
        ax2.set_title('Noisy Image')
        ax2.axis('off')
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)

        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(img_denoised)
        ax3.set_title('Denoised Image')
        ax3.axis('off')
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax3.add_patch(rect)

        # Plot detail regions in second row
        ax4 = plt.subplot(gs[1, 0])
        ax4.imshow(region_orig)
        ax4.set_title('Original Detail')
        ax4.axis('off')

        ax5 = plt.subplot(gs[1, 1])
        ax5.imshow(region_noisy)
        ax5.set_title('Noisy Detail')
        ax5.axis('off')

        ax6 = plt.subplot(gs[1, 2])
        ax6.imshow(region_denoised)
        ax6.set_title('Denoised Detail')
        ax6.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

    except Exception as e:
        print(f"Error during detail visualization: {str(e)}")
        plt.close()
