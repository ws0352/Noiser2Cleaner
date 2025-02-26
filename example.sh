#!/bin/bash
# 示例使用脚本

# 确保目录存在
mkdir -p checkpoints
mkdir -p results

# 示例 1: 训练模型
echo "===== 开始训练模型 ====="
python main.py train --image_path sample.png --epochs 50 --batch_size 16 --patch_size 128 --num_samples 2000

# 示例 2: 评估模型
echo "===== 开始评估模型 ====="
python main.py eval --image_path test.png --checkpoint_path checkpoints/best_model.pth --output_dir results

# 示例 3: 只使用评估模式处理另一张图像
echo "===== 处理另一张图像 ====="
python main.py eval --image_path another_image.png --checkpoint_path checkpoints/best_model.pth --output_dir results/another_image

# 示例 4: 批量处理文件夹中的所有图像
echo "===== 批量处理图像 ====="
mkdir -p results/batch
for img in input_images/*.png; do
    filename=$(basename -- "$img")
    echo "处理图像: $filename"
    python main.py eval --image_path "$img" --checkpoint_path checkpoints/best_model.pth --output_dir "results/batch/${filename%.*}"
done

echo "所有处理完成，结果保存在 results 目录"