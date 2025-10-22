# %%
import os
import cv2
from jittor import jt
from jittor.dataset import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from jittor import transform as transforms
from PIL import Image 
from jittor.dataset import DataLoader 

# --- 配置参数 ---

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 修改数据集路径为绝对路径
DATASET_PATH = os.path.join(project_root, 'tomato_yolo_dataset')
# 训练集图片和标签路径
IMAGE_Train_PATH = os.path.join(DATASET_PATH, 'images', 'train')
LABEL_Train_PATH = os.path.join(DATASET_PATH, 'labels', 'train')

# 类别名称字典，各种番茄的病症名称缩写

CLASS_NAMES = {0: 'TMBS',
              1: 'TEB',
              2:'TLB',
              3:'TLM',
              4:'TSLS',
              5:'TSM',
              6:'TTS',
              7:'TYLCV',
              8:'TMV',
              9:'TH'}



# %%
from config import Config
config = Config()

# %%
class Tomato_DataSet(Dataset):

    #image_dir:图片路径是tomato_yolo_dataset/images/train
    #label_dir:标签路径是tomato_yolo_dataset/labels/train
    #image_files:是image_dir下的所有图片名称组成的列表
    def __init__(self, img_dir,label_dir, transform=None,img_size=config.IMG_SIZE):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size

        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png','.jpeg')) ]
        self.set_attrs(total_len=len(self.img_files))

    def __getitem__(self,idx):
        try:
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir,img_name)
            img =Image.open(img_path).convert('RGB')

            #读取标签
            label_path = os.path.join(self.label_dir,img_name.rsplit('.',1)[0]+'.txt')
            boxes = [] #储存边界框
            label = [] #储存类别标签
            if os.path.exists(label_path):
                with open(label_path,'r') as f:
                    for line in f.readlines():
                        #YOLO格式
                        class_id,x_center,y_center,width,height = map(float,line.strip().split())
                        
                        img_w,img_h = img.size
                        x_center = x_center * img_w
                        y_center = y_center * img_h
                        width = width * img_w
                        height = height * img_h

                        x1 = x_center - width/2
                        y1 = y_center - height/2
                        x2 = x_center + width/2
                        y2 = y_center + height/2

                        boxes.append([x1,y1,x2,y2])
                        label.append(int(class_id))
            
            if self.transform:
                img = self.transform(img)  # 应用Resize和ToTensor

                #使用ImageNet的均值和方差进行标准化
                mean= jt.array([0.485, 0.456, 0.406]).view(3,1,1)
                std = jt.array([0.229, 0.224, 0.225]).view(3,1,1)
                img = (img - mean) / std
            
            boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), dtype=np.float32)
            label = np.array(label, dtype=np.int64) if label else np.zeros((0,), dtype=np.int64)
        
            return img, boxes, label
            
        except Exception as e:
            print(f"Error loading {img_name}: {str(e)}")
            # 返回一个空的样本（用于错误处理）
            empty_img = jt.zeros((3, self.img_size, self.img_size))
            empty_boxes = np.zeros((0, 4), dtype=np.float32)
            empty_labels = np.zeros((0,), dtype=np.int64)
            return empty_img, empty_boxes, empty_labels
    def create_dataloader(
                    data_root_dir,# path to data,路径是tomato_yolo_dataset
                    images_train_dir = 'images/train',
                    labels_train_dir = 'labels/train',
                    images_val_dir = 'images/val',
                    labels_val_dir = 'labels/val',
                    batch_size=config.BATCH_SIZE,
                    img_size=224,
                    is_train = True):
        """
        Args:
            data_root_dir: 数据集根目录
            batch_size: 批次大小
            img_size: 图片尺寸
            is_train: 是否为训练模式
        
        Return:
            dataloader: 数据加载器
        """
        try:
            if is_train:
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(0.5),  # 随机水平翻转
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ])
            images_dir = os.path.join(data_root_dir , images_train_dir if is_train else images_val_dir)
            labels_dir = os.path.join(data_root_dir , labels_train_dir if is_train else labels_val_dir)

            dataset = Tomato_DataSet(
                img_dir = images_dir,
                label_dir = labels_dir,
                transform = transform,
                img_size = img_size
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=is_train,
                num_workers=0,
                drop_last=is_train
            )
            
            return dataloader

        except Exception as e:
            print(f"Error creating dataloader: {str(e)}")
            return None
            
            



# %%
