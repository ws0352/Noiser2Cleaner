import os
import argparse
import torch
import numpy as np

from models import ImprovedUNet
from data import ImprovedDataset
from train import main as train_main
from evaluate import main as evaluate_main


def setup_env():
    """设置环境参数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 确保目录存在
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Image Denoising')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # 训练模式参数
    train_parser = subparsers.add_parser('train', help='Train a denoising model')
    train_parser.add_argument('--image_path', type=str, required=True, help='Path to the clean image')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--patch_size', type=int, default=128, help='Size of image patches')
    train_parser.add_argument('--num_samples', type=int, default=2000, help='Number of patches to extract')

    # 评估模式参数
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('--image_path', type=str, required=True, help='Path to the test image')
    eval_parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')

    # 解析参数
    args = parser.parse_args()

    # 设置环境
    setup_env()

    # 根据模式执行操作
    if args.mode == 'train':
        print("=== Training Mode ===")
        train_main(
            image_path=args.image_path,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            num_samples=args.num_samples
        )
    elif args.mode == 'eval':
        print("=== Evaluation Mode ===")
        evaluate_main(
            image_path=args.image_path,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
