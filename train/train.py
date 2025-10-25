import os
import sys
import jittor as jt
from jittor import nn
import traceback
import numpy as np
from datetime import timedelta


current_dir = os.path.abspath('.')
project_root = os.path.dirname(current_dir)

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)

from data_loader import get_dataloader, CLASS_NAMES
from models.vit_model import Visual_Transformer
from config import Config

from tqdm import tqdm
import time

#设置随机数种子
def set_seed(seed = 42):
    np.random.seed(seed)

#计算准确率，outputs是 B*C张量，labels是 B张量
def calculate_accuracy(outputs,labels):
    preds = jt.argmax(outputs,dim=1)[0]#返回的是最大概率的索引，B张量，没有保持维度
    correct = jt.sum(preds == labels)

    return float(correct) / labels.shape[0]

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def validate_labels(labels, num_classes):
    """快速标签验证"""
    labels_np = labels.numpy()
    valid_mask = (labels_np >= 0) & (labels_np < num_classes)
    return valid_mask


def calculate_balanced_accuracy(outputs, labels, num_classes):
    """
    计算平衡准确率（Balanced Accuracy）
    """
    preds = jt.argmax(outputs, dim=1)[0]
    labels_np = labels.numpy()
    preds_np = preds.numpy()
    
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(num_classes):
        class_mask = (labels_np == i)
        if class_mask.sum() > 0:
            class_correct = (preds_np[class_mask] == i).sum()
            class_acc = class_correct / class_mask.sum()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    # 返回平均值（平衡准确率）
    return sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0


def calculate_per_class_accuracy(outputs, labels, num_classes, class_names):
    """
    计算每个类别的准确率
    """
    preds = jt.argmax(outputs, dim=1)[0]
    labels_np = labels.numpy()
    preds_np = preds.numpy()
    
    per_class_acc = {}
    for i in range(num_classes):
        class_mask = (labels_np == i)
        if class_mask.sum() > 0:
            class_correct = (preds_np[class_mask] == i).sum()
            class_acc = class_correct / class_mask.sum()
            per_class_acc[class_names[i]] = class_acc
        else:
            per_class_acc[class_names[i]] = 0.0
    
    return per_class_acc


def train():
    config = Config()
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    set_seed(42)

    data_root = os.path.join(project_root, 'tomato_yolo_dataset')
    save_dir = os.path.join(project_root,'train', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    train_loader = get_dataloader(
        root_dir=data_root,
        mode='train',
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        sample_ratio=0.1
    )

    val_loader = get_dataloader(
        root_dir=data_root,
        mode='val',
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        shuffle=False,  # ✅ 验证集不shuffle
        num_workers=config.NUM_WORKERS,
        sample_ratio=0.1
    )

    # ✅ 先统计类别分布
    print("\n检查数据集类别分布...")
    class_counts = np.zeros(len(CLASS_NAMES))
    for _, labels in train_loader:
        labels_np = labels.numpy()
        for i in range(len(CLASS_NAMES)):
            class_counts[i] += (labels_np == i).sum()
    
    print("训练集类别分布:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {int(class_counts[i])} ({class_counts[i]/class_counts.sum()*100:.1f}%)")
    
    # ✅ 计算类别权重（解决不平衡问题）
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(CLASS_NAMES) * class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)  # 归一化
    class_weights = jt.array(class_weights, dtype=jt.float32)
    print(f"\n类别权重: {class_weights}")

    model = Visual_Transformer(
        img_size=config.IMG_SIZE, 
        patch_size=config.PATCH_SIZE,
        in_channels=config.IN_CHANNELS, 
        embed_dim=config.EMBED_DIM,
        depth=config.NUM_LAYERS, 
        num_heads=config.NUM_HEADS,
        dropout_rate=0.1, 
        hidden_dim=config.MLP_Hidden_Dim
    )

    print("\n验证模型初始化...")
    model.eval()
    with jt.no_grad():
        test_img, _ = next(iter(train_loader))
        test_out = model(test_img[:2])
        print(f"✓ 前向传播成功: shape={test_out.shape}")

    # ✅ 使用带权重的优化器
    optimizer = nn.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    # ✅ 添加学习率调度器
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    best_acc = 0.0
    best_balanced_acc = 0.0
    
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_start_time = time.time()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batches = 0  # ✅ 单独计数
        
        # ✅ 正确使用进度条
        pbar = tqdm(enumerate(train_loader), 
                    total=len(train_loader), 
                    desc=f'Epoch {epoch+1}/{config.EPOCHS}')

        for batch_idx, (images, labels) in pbar:
            # 验证标签
            valid_mask = validate_labels(labels, len(CLASS_NAMES))
            
            if not valid_mask.all():
                invalid_labels = labels.numpy()[~valid_mask]
                print(f"\n⚠️  Batch {batch_idx} 无效标签: {invalid_labels}")
                
                valid_indices = jt.array(valid_mask)
                if valid_indices.sum() == 0:
                    continue
                images = images[valid_indices]
                labels = labels[valid_indices]

            # 前向传播
            outputs = model(images)
            
            # ✅ 使用加权损失
            loss = nn.cross_entropy_loss(outputs, labels, weight=class_weights)
            
            # 反向传播
            optimizer.step(loss)
            
            # 计算准确率
            preds = jt.argmax(outputs, dim=1)[0]
            correct = jt.sum(preds == labels).item()
            
            train_loss += loss.item()
            train_correct += correct
            train_total += labels.shape[0]
            train_batches += 1  # ✅ 实际处理的batch数
            
            # ✅ 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/labels.shape[0]:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # ✅ 更新学习率
        scheduler.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        all_val_outputs = []
        all_val_labels = []

        with jt.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                valid_mask = validate_labels(labels, len(CLASS_NAMES))
                
                if not valid_mask.all():
                    valid_indices = jt.array(valid_mask)
                    if valid_indices.sum() == 0:
                        continue
                    images = images[valid_indices]
                    labels = labels[valid_indices]

                outputs = model(images)
                loss = nn.cross_entropy_loss(outputs, labels, weight=class_weights)
                
                val_loss += loss.item()
                preds = jt.argmax(outputs, dim=1)[0]
                correct = jt.sum(preds == labels).item()
                
                val_correct += correct
                val_total += labels.shape[0]
                val_batches += 1
                
                all_val_outputs.append(outputs)
                all_val_labels.append(labels)

        # ✅ 修正平均值计算
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        avg_train_acc = train_correct / train_total if train_total > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        avg_val_acc = val_correct / val_total if val_total > 0 else 0

        if all_val_outputs and all_val_labels:
            all_outputs = jt.concat(all_val_outputs, dim=0)
            all_labels = jt.concat(all_val_labels, dim=0)
            balanced_val_acc = calculate_balanced_accuracy(all_outputs, all_labels, len(CLASS_NAMES))
            per_class_acc = calculate_per_class_accuracy(all_outputs, all_labels, len(CLASS_NAMES), CLASS_NAMES)
        else:
            balanced_val_acc = 0.0
            per_class_acc = {}

        epoch_time = time.time() - epoch_start_time

        # 打印结果
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.EPOCHS} - {format_time(epoch_time)}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Balanced Acc: {balanced_val_acc:.4f}")

        if per_class_acc:
            print("\nPer-class accuracy:")
            for class_name, acc in per_class_acc.items():
                print(f"  {class_name:25s}: {acc:.4f}")

        # 保存模型
        if balanced_val_acc > best_balanced_acc:
            best_balanced_acc = balanced_val_acc
            model_path = os.path.join(save_dir, 'best_balanced_model.pkl')
            jt.save(model.state_dict(), model_path)
            print(f"\n✓ 新的最佳平衡准确率模型已保存: {best_balanced_acc:.4f}")

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            model_path = os.path.join(save_dir, 'best_model.pkl')
            jt.save(model.state_dict(), model_path)
            print(f"✓ 新的最佳准确率模型已保存: {best_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"最佳平衡准确率: {best_balanced_acc:.4f}")

if __name__ == '__main__':
    train()


