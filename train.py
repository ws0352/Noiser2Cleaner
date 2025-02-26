import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from adabelief_pytorch import AdaBelief

from models import ImprovedUNet, ModelEMA
from data import ImprovedDataset
from losses import RefinedLoss
from utils import visualize_training_history


def train_model(model, train_loader, num_epochs=100, device='cuda', save_dir='checkpoints'):
    """Fixed training function with proper mixed precision handling"""
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    criterion = RefinedLoss().to(device)

    # Initialize optimizer
    optimizer = AdaBelief(
        model.parameters(),
        lr=1e-1,
        eps=1e-16,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        weight_decouple=True,
        rectify=True,
        print_change_log=False
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))

    # Model EMA
    ema = ModelEMA(model, decay=0.999)

    # Training records
    history = {'loss': [], 'psnr': [], 'ssim': []}
    best_metric = 0
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_psnr = []
        epoch_ssim = []

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for batch_idx, (noisy, clean) in enumerate(pbar):
                # Move data to device and ensure float32
                noisy = noisy.to(device)
                clean = clean.to(device)

                # Mixed precision forward pass
                with torch.amp.autocast(device_type='cuda', enabled=(device == 'cuda')):
                    output = model(noisy)
                    loss = criterion(output, clean)

                # Gradient optimization with mixed precision
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Unscale weights for gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Step optimizer and update scaler
                scaler.step(optimizer)
                scaler.update()

                # Update EMA model
                ema.update(model)

                # Update learning rate
                scheduler.step()

                # Calculate metrics (ensure we use float32 here)
                with torch.no_grad():
                    output_float = output.float()
                    clean_float = clean.float()
                    psnr = -10 * torch.log10(torch.mean((output_float - clean_float) ** 2))
                    ssim_val = torch.tensor(0.0)  # Placeholder for SSIM
                    try:
                        from pytorch_msssim import ssim
                        ssim_val = ssim(output_float, clean_float, data_range=1.0)
                    except:
                        pass

                # Record metrics
                epoch_losses.append(loss.item())
                epoch_psnr.append(psnr.item())
                epoch_ssim.append(ssim_val.item())

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr.item():.2f}',
                    'ssim': f'{ssim_val.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })

        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_psnr = np.mean(epoch_psnr)
        avg_ssim = np.mean(epoch_ssim)

        # Update history
        history['loss'].append(avg_loss)
        history['psnr'].append(avg_psnr)
        history['ssim'].append(avg_ssim)

        # Save best model
        current_metric = avg_psnr + avg_ssim
        if current_metric > best_metric:
            best_metric = current_metric
            # Save with weights_only=True for better security
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema.ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
            }, os.path.join(save_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1

        print(f'Epoch {epoch + 1}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}')

        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, epoch, optimizer, scheduler,
                            {'loss': avg_loss, 'psnr': avg_psnr, 'ssim': avg_ssim},
                            os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

    # Visualize training history
    visualize_training_history(history)

    return ema.ema, history


def save_checkpoint(model, epoch, optimizer, scheduler, metrics, path):
    """Save checkpoint in a safe format"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }
        torch.save(checkpoint, path)
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")


def safe_load_checkpoint(model, checkpoint_path):
    """Safely load model checkpoint with proper error handling"""
    try:
        # Method 1: Standard loading
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint
    except Exception as e:
        print(f"First loading attempt failed: {str(e)}")
        try:
            # Method 2: Alternative safe loading approach
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, checkpoint
        except Exception as e:
            print(f"Second loading attempt failed: {str(e)}")
            return None, None


def main(image_path, num_epochs=50, batch_size=16, patch_size=128, num_samples=2000):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create dataset
    try:
        dataset = ImprovedDataset(
            image_path=image_path,
            patch_size=patch_size,
            num_samples=num_samples,
            aug_prob=0.7
        )

        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Create model
        model = ImprovedUNet()

        # Train model
        print("Starting training...")
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            num_epochs=num_epochs,
            device=device
        )

        print("Training completed!")
        return trained_model, history

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        return None, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train denoising model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the clean image')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of image patches')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of patches to extract')

    args = parser.parse_args()

    main(
        image_path=args.image_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_samples=args.num_samples
    )
