
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
