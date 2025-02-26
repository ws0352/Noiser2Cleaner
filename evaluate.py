import os
import torch
import numpy as np

from models import ImprovedUNet
from data import ImprovedDataset
from utils import (
    load_and_process_image,
    denoise_image,
    evaluate_metrics,
    compute_texture_score,
    visualize_results,
    visualize_comparative_detail
)
from train import safe_load_checkpoint


def evaluate_model(model, image_path, device='cuda', output_dir='results'):
    """Evaluate a trained denoising model on a test image"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load original image
        original = load_and_process_image(image_path)
        if original is None:
            print("Failed to load test image")
            return

        # Create dataset instance for noise generation
        dataset = ImprovedDataset(
            image_path=image_path,
            patch_size=128,
            num_samples=25,  # Small number just for initialization
            aug_prob=0.7
        )

        # Generate noisy version
        noisy = dataset.add_realistic_noise(original)

        # Process with the model
        print("Denoising image...")
        model.eval()
        denoised = denoise_image(
            model=model,
            image_path=image_path,  # We pass the path since denoise_image loads it internally
            device=device
        )

        # Save results
        print("Saving results...")
        cv2_save_path = os.path.join(output_dir, 'denoised.png')

        # Convert to uint8 for saving
        import cv2
        cv2.imwrite(
            cv2_save_path,
            cv2.cvtColor((denoised * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )

        # Calculate metrics
        metrics = evaluate_metrics(original, denoised)
        texture_score = compute_texture_score(original, denoised)
        metrics['Texture_Score'] = texture_score

        # Print metrics
        print("\nDenoising Results:")
        print(f"PSNR: {metrics['PSNR']:.2f} dB")
        print(f"SSIM: {metrics['SSIM']:.4f}")
        print(f"Texture Preservation: {metrics['Texture_Score']:.4f}")

        # Visualize results
        visualize_results(
            original=original,
            noisy=noisy,
            denoised=denoised,
            save_path=os.path.join(output_dir, 'comparison.png')
        )

        # Visualize detailed comparison
        visualize_comparative_detail(
            original=original,
            noisy=noisy,
            denoised=denoised,
            save_path=os.path.join(output_dir, 'detail_comparison.png')
        )

        print(f"\nResults saved to {output_dir}")
        return metrics

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None


def main(image_path, checkpoint_path, output_dir='results'):
    """Main evaluation function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = ImprovedUNet()

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model, checkpoint_info = safe_load_checkpoint(model, checkpoint_path)

    if model is None:
        print("Failed to load model checkpoint")
        return

    # Move model to device
    model = model.to(device)

    # Evaluate
    print(f"Evaluating on {image_path}")
    metrics = evaluate_model(model, image_path, device, output_dir)

    if metrics:
        # If checkpoint contains training info, print it
        if checkpoint_info and 'epoch' in checkpoint_info:
            print(f"\nModel trained for {checkpoint_info['epoch'] + 1} epochs")
            if 'metrics' in checkpoint_info:
                print(f"Training metrics at checkpoint:")
                for k, v in checkpoint_info['metrics'].items():
                    print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate denoising model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the test image')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')

    args = parser.parse_args()

    main(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir
    )
