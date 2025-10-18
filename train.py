
import sys
import os
import time
import argparse

current_dir = os.path.abspath('.')
project_root = current_dir

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)

import jittor as jt
from jittor import nn
import numpy as np
from tqdm import tqdm

from config import Config
from models.vit_model import Visual_Transformer
from data_loader import get_data_loaders, create_dummy_data_loaders

config = Config()

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs, targets

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计信息
        running_loss += loss.item()
        _, predicted = jt.argmax(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with jt.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation'):
            inputs, targets = inputs, targets

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计信息
            running_loss += loss.item()
            _, predicted = jt.argmax(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc

def train_model(args):
    # 设置设备
    jt.flags.use_cuda = jt.has_cuda

    # 创建模型
    model = Visual_Transformer()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = jt.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = jt.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 加载数据
    if args.use_dummy_data:
        print("使用虚拟数据进行训练...")
        train_loader, val_loader = create_dummy_data_loaders(batch_size=args.batch_size, num_batches=20)
    else:
        print(f"从 {args.train_dir} 加载训练数据...")
        train_loader, val_loader = get_data_loaders(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size
        )

    # 训练模型
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        print('-' * 10)

        # 训练阶段
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, None)

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, None)

        # 更新学习率
        scheduler.step()

        # 打印统计信息
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            jt.save(model.state_dict(), best_model_path)
            print(f'保存最佳模型到 {best_model_path}')

    print(f'训练完成! 最佳验证准确率: {best_val_acc:.2f}%')

    return model

def main():
    parser = argparse.ArgumentParser(description='训练ViT模型')
    parser.add_argument('--train_dir', type=str, default='data/train', help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default='data/val', help='验证数据目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='权重衰减')
    parser.add_argument('--use_dummy_data', action='store_true', help='使用虚拟数据进行训练')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 开始训练
    model = train_model(args)

if __name__ == '__main__':
    main()
