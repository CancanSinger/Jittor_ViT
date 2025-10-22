import jittor as jt
from jittor import nn, optim
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.abspath('.')
project_root = os.path.dirname(current_dir)

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)

from data_loader import Tomato_DataSet
from models.vit_model import Visual_Transformer
from config import Config

# 初始化配置
config = Config()

def train_model():
    # 创建模型
    model = Visual_Transformer()
    
    # 创建交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
    
    # 创建数据加载器
    train_loader = Tomato_DataSet.create_dataloader(
        data_root_dir='tomato_yolo_dataset',
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        is_train=True
    )
    
    val_loader = Tomato_DataSet.create_dataloader(
        data_root_dir='tomato_yolo_dataset',
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        is_train=False
    )
    
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")
    
    # 训练循环
    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, boxes, labels) in enumerate(train_loader):
            # 调试信息：查看批次数据结构
            if i == 0 and epoch == 0:
                print(f"批次数据形状:")
                print(f"  images.shape: {images.shape}")
                print(f"  boxes类型: {type(boxes)}")
                print(f"  labels类型: {type(labels)}")
                print(f"  labels长度: {len(labels)}")
                if len(labels) > 0:
                    print(f"  第一个标签样本形状: {labels[0].shape}")
            
            # 处理标签 - 获取每个样本的标签
            target_labels = []
            for j in range(len(labels)):  # 遍历批次中的每个样本
                label_batch = labels[j]  # 获取第j个样本的标签
                if len(label_batch) > 0:
                    # 如果一张图像中有多个对象，选择第一个对象的类别
                    target_labels.append(int(label_batch[0]))  # label_batch[0]是类别ID
                else:
                    # 如果没有标签，设置为默认类别（如0）
                    target_labels.append(0)
            
            target_labels = jt.array(target_labels).long()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, target_labels)
            
            # 反向传播和优化
            optimizer.step(loss)
            
            # 统计训练准确率
            _, predicted = jt.argmax(outputs.data, 1)
            train_total += target_labels.size(0)
            train_correct += (predicted == target_labels).sum().item()
            train_loss += loss.item()
            
            # 定期打印训练信息
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with jt.no_grad():
            for images, boxes, labels in val_loader:
                # 处理标签
                target_labels = []
                for j in range(len(labels)):
                    label_batch = labels[j]
                    if len(label_batch) > 0:
                        target_labels.append(int(label_batch[0]))
                    else:
                        target_labels.append(0)
                
                target_labels = jt.array(target_labels).long()
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, target_labels)
                
                # 统计验证准确率
                _, predicted = jt.argmax(outputs.data, 1)
                val_total += target_labels.size(0)
                val_correct += (predicted == target_labels).sum().item()
                val_loss += loss.item()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
    
    # 保存模型
    model.save('vit_tomato_disease_model.pkl')
    print("模型已保存为 'vit_tomato_disease_model.pkl'")

# 简化版本的训练函数，更容易理解批次处理
def train_model_simple():
    # 创建模型、损失函数和优化器
    model = Visual_Transformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 获取当前文件所在目录的父目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_root = os.path.join(project_root, 'tomato_yolo_dataset')
    
    # 检查数据集路径是否存在
    if not os.path.exists(data_root):
        print(f"数据集路径不存在: {data_root}")
        return
        
    print(f"使用数据集路径: {data_root}")
    
    # 创建数据加载器
    try:
        train_loader = Tomato_DataSet.create_dataloader(
            data_root_dir=data_root,  # 使用绝对路径
            batch_size=config.BATCH_SIZE,
            img_size=config.IMG_SIZE,
            is_train=True
        )
        
        if train_loader is None:
            print("数据加载器创建失败")
            return
            
        print("数据加载器创建成功")
        print("开始训练...")
        
        # 训练循环
        for i, (images, boxes, labels) in enumerate(train_loader):
            print(f"批次 {i+1}:")
            print(f"  图像张量形状: {images.shape}")
            print(f"  标签列表长度: {len(labels)}")
            
            # 处理标签
            batch_labels = []
            for j in range(len(labels)):
                if len(labels[j]) > 0:
                    class_id = int(labels[j][0])  # 获取类别ID
                    batch_labels.append(class_id)
                else:
                    batch_labels.append(0)  # 默认类别
            
            target_labels = jt.array(batch_labels).long()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, target_labels)
            
            # 反向传播
            optimizer.step(loss)
            
            print(f"  损失值: {loss.item():.4f}")
            
            # 只处理前几个批次作为示例
            if i >= 2:
                break
                
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("训练示例完成")

if __name__ == "__main__":
    train_model_simple()