* # 零样本图像去噪项目

    这是一个基于PyTorch的零样本图像去噪项目，使用单张干净图像训练模型，可以对同类型的图像进行高质量的去噪处理。

    ## 特性

    - **零样本学习**: 仅使用单张干净图像进行训练
    - **高级网络结构**: 使用改进的UNet架构，集成了小波变换和细节保留模块
    - **高质量去噪**: 专注于细节和纹理保留的去噪算法
    - **自适应噪声模拟**: 模拟各种真实的噪声类型，包括高斯噪声、泊松噪声、椒盐噪声等
    - **智能分块处理**: 使用图块加权融合技术，可处理任意大小的图像

    ## 项目结构

    ```
    denoise_project/
    ├── models/              # 网络模型定义
    ├── losses/              # 损失函数
    ├── data/                # 数据集处理
    ├── utils/               # 工具函数
    ├── train.py             # 训练主函数
    ├── evaluate.py          # 评估主函数
    └── main.py              # 主入口点
    ```

    ## 安装依赖

    ```bash
    pip install torch torchvision pytorch-msssim opencv-python numpy tqdm adabelief-pytorch matplotlib scikit-image
    ```

    ## 使用方法

    ### 训练模式

    使用一张干净图像训练去噪模型：

    ```bash
    python main.py train --image_path path/to/clean/image.png --epochs 50 --batch_size 16
    ```

    可选参数：
    - `--patch_size`: 训练用图像块大小 (默认: 128)
    - `--num_samples`: 从图像中提取的patch数量 (默认: 2000)

    ### 评估模式

    使用训练好的模型对图像进行去噪：

    ```bash
    python main.py eval --image_path path/to/test/image.png --checkpoint_path checkpoints/best_model.pth
    ```

    可选参数：
    - `--output_dir`: 指定输出目录 (默认: "results")

    ## 工作原理

    1. **数据生成**: 从单张干净图像中提取多个图像块，并自动添加各种类型的噪声
    2. **自适应训练**: 使用精细的损失函数训练模型，保持图像细节和纹理
    3. **模型评估**: 使用PSNR、SSIM和纹理保留评分等指标评估去噪效果
    4. **结果可视化**: 提供原图、噪声图和去噪结果的对比，以及关键细节区域的放大对比

    ## 高级特性

    - **梯度引导采样**: 智能提取包含更多细节/边缘的区域用于训练
    - **小波域处理**: 在小波域进行特征提取，更好地保留细节
    - **混合精度训练**: 自动使用CUDA混合精度加速训练
    - **指数移动平均**: 使用EMA技术提高模型稳定性和性能

    ## 结果示例

    去噪结果将保存在`results`目录，包括：
    - `denoised.png`: 去噪后的完整图像
    - `comparison.png`: 原图、噪声图和去噪结果的并排比较
    - `detail_comparison.png`: 关键区域的放大对比

    ## 许可证

    MIT License
