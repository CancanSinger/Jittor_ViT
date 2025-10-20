
# Vision Transformer (ViT) 模型实现

本项目实现了基于Jittor框架的Vision Transformer模型，已经搭建好基本的骨架，下一步需要进行检测头的搭建

## 项目结构

```
ViT/
├── config.py                # 配置参数
├── data_loader.py           # 数据加载器
├── inference.py             # 推理脚本
├── main.py                  # 主脚本
├── train.py                 # 训练脚本
├── models/                  # 模型实现
│   ├── vit_model.py         # ViT模型
│   └── Layers/              # 模型层实现
│       ├── attention.py     # 多头自注意力机制
│       ├── MLP.py           # 多层感知机
│       ├── layer_norm.py    # 层归一化
│       └── patch_embed.py   # 图像块嵌入
└── outputs/                 # 输出目录
    └── best_model.pth       # 训练好的模型权重
```

## 安装依赖

```bash
pip install jittor
pip install numpy
pip install pillow
pip install tqdm
```

## 使用方法

### 1. 训练模型

#### 使用虚拟数据训练（快速测试）

```bash
python main.py train --use_dummy_data --epochs 5 --batch_size 16
```

#### 使用真实数据训练

```bash
python main.py train --train_dir data/train --val_dir data/val --epochs 10 --batch_size 32
```

### 2. 模型推理

#### 单张图像推理

```bash
python main.py inference --model_path outputs/best_model.pth --input path/to/image.jpg --mode single
```

#### 目录批量推理

```bash
python main.py inference --model_path outputs/best_model.pth --input path/to/images --mode directory
```

#### 模型评估

```bash
python main.py inference --model_path outputs/best_model.pth --input data/val --mode evaluate
```

### 3. 完整流程（训练+推理）

```bash
# 使用虚拟数据
python main.py full --use_dummy_data --epochs 5

# 使用真实数据
python main.py full --train_dir data/train --val_dir data/val --epochs 10
```

## 参数说明

### 训练参数

- `--train_dir`: 训练数据目录
- `--val_dir`: 验证数据目录
- `--output_dir`: 输出目录（默认为outputs）
- `--epochs`: 训练轮数（默认为10）
- `--batch_size`: 批次大小（默认为32）
- `--lr`: 学习率（默认为1e-4）
- `--weight_decay`: 权重衰减（默认为5e-2）
- `--use_dummy_data`: 使用虚拟数据进行训练

### 推理参数

- `--model_path`: 模型权重路径
- `--input`: 输入图像或目录路径
- `--output_dir`: 输出目录（默认为inference_results）
- `--mode`: 推理模式（single/directory/evaluate）
- `--data_dir`: 数据目录（用于评估模式）
- `--class_names`: 类别名称列表

## 模型配置

模型配置在`config.py`中定义：

```python
class Config:
    def __init__(self):
        # 图像相关配置
        self.IMG_SIZE = 224          # 输入图像尺寸
        self.IN_CHANNELS = 3         # 输入通道数（RGB图像）
        self.PATCH_SIZE = 16         # 图像块大小
        self.EMBED_DIM = 768         # 嵌入维度

        self.DROPOUT = 0.1           # dropout概率
        self.NUM_LAYERS = 12         # Transformer层数
        self.NUM_HEADS = 12          # 注意力头数
        self.MLP_Hidden_Dim = 768*4  # MLP隐藏层维度
        self.MLP_RATIO = 4           # MLP扩展比例
        self.NUM_CLASSES = 80        # 分类任务的类别数
```

## 示例

### 快速测试（使用虚拟数据）

```bash
# 训练模型
python main.py train --use_dummy_data --epochs 3 --batch_size 8

# 推理
python main.py inference --model_path outputs/best_model.pth --input dummy_image.jpg --mode single
```

### 完整流程（使用虚拟数据）

```bash
python main.py full --use_dummy_data --epochs 3 --batch_size 8
```

## 注意事项

1. 如果没有真实数据，可以使用`--use_dummy_data`参数进行快速测试。
2. 模型权重默认保存在`outputs/best_model.pth`。
3. 推理结果默认保存在`inference_results/`目录下。
4. 确保输入图像大小与配置中的`IMG_SIZE`一致，或让脚本自动调整大小。
