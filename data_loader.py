
import sys
import os

current_dir = os.path.abspath('.')
project_root = current_dir

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)

import jittor as jt
from jittor import dataset
from jittor.transform import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from config import Config
config = Config()

class CustomDataset(dataset.Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        # 这里应该根据你的数据集结构进行修改
        # 示例代码，假设数据集按文件夹组织，每个文件夹代表一个类别
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_dir, file_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        # 加载图像
        img = jt.data.load(path)

        # 应用变换
        if self.transform:
            img = self.transform(img)

        return img, target

def get_data_loaders(train_dir, val_dir=None, batch_size=32, num_workers=4):
    # 训练数据变换
    train_transform = Compose([
        Resize((config.IMG_SIZE, config.IMG_SIZE)),
        RandomHorizontalFlip(),
        RandomCrop(config.IMG_SIZE, padding=4),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试数据变换
    val_transform = Compose([
        Resize((config.IMG_SIZE, config.IMG_SIZE)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = CustomDataset(train_dir, mode='train', transform=train_transform)

    if val_dir:
        val_dataset = CustomDataset(val_dir, mode='val', transform=val_transform)
    else:
        # 如果没有提供验证集，从训练集中划分一部分作为验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = jt.dataset.random_split(train_dataset, [train_size, val_size])
        val_dataset.transform = val_transform

    # 创建数据加载器
    train_loader = dataset.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = dataset.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

# 创建一个简单的数据加载器用于测试
def create_dummy_data_loaders(batch_size=4, num_batches=10):
    # 创建虚拟数据用于测试
    train_data = []
    val_data = []

    for _ in range(num_batches):
        # 训练数据
        train_batch = jt.randn(batch_size, 3, config.IMG_SIZE, config.IMG_SIZE)
        train_labels = jt.randint(0, config.NUM_CLASSES, (batch_size,))
        train_data.append((train_batch, train_labels))

        # 验证数据
        val_batch = jt.randn(batch_size, 3, config.IMG_SIZE, config.IMG_SIZE)
        val_labels = jt.randint(0, config.NUM_CLASSES, (batch_size,))
        val_data.append((val_batch, val_labels))

    return train_data, val_data
