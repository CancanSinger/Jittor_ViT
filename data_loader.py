import os
import jittor as jt
from jittor.dataset import Dataset
from PIL import Image #pillow
import numpy as np

#prefix
PREFIX_TO_CLASS = {
    'TMBS': 0, 'TEB': 1, 'TLB': 2, 'TLM': 3, 
    'TSLS': 4, 'TMSL': 4,
    'TSM': 5, 'TTS': 6, 'TYLCV': 7, 'TMV': 8, 'TH': 9
}

CLASS_NAMES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
    'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot',
    'Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'Healthy'
]

class TomatoDataset(Dataset):
    def __init__(self,root_dir,mode = 'train',img_size=224):
        super().__init__()
        self.img_size = img_size
        self.mode = mode
        self.root_dir = root_dir
        
        self.img_dir = os.path.join(root_dir, 'images', mode)
        self.label_dir = os.path.join(root_dir, 'labels', mode)
        
        self.samples = []
        self._load_data()#把数据放到samples列表中，这里相当于直接执行了_load_data函数
    def _load_data(self):
        """加载YOLO格式数据"""
        #检查图片路径
        if not os.path.exists(self.img_dir):
            raise ValueError(f"图片目录不存在: {self.img_dir}")
        
        #检查标签目录
        if not os.path.exists(self.label_dir):
            raise ValueError(f"标签目录不存在: {self.label_dir}")
        
        #遍历这个文件夹中的所有文件
        for filename in os.listdir(self.img_dir):
            if not filename.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                continue

            img_path = os.path.join(self.img_dir, filename)

            label_path = os.path.join(self.label_dir, filename.rsplit('.', 1)[0] + '.txt')

            if os.path.exists(label_path):
                try:
                    with open(label_path,'r') as f:
                        line = f.readline().strip()
                        if line:
                            label = int(line.split()[0])
                            self.samples.append((img_path,label))
                            continue
                except Exception as e:
                    print(f"警告: 无法读取标签文件 {label_path}: {e}")

        #如果没有标签文件，就推断标签
        prefix = filename.split('_')[0]
        if prefix in PREFIX_TO_CLASS:
            label = PREFIX_TO_CLASS[prefix]
            self.samples.append((img_path,label))
    
    def _print_stats(self):
      
        if len(self.samples) == 0:
            print("  ⚠️  没有找到任何样本！")
            return
        
        # 初始化计数器
        counts = [0] * 10
        
        # 统计每个类别的样本数
        for _, label in self.samples:
            counts[label] += 1
        
        # 打印统计信息
        print(f"  类别分布:")
        for i, count in enumerate(counts):
            if count > 0:
                print(f"    类别{i} ({CLASS_NAMES[i]}): {count}")


    def __getitem__(self,idx):#根据索引获取图片路径和标签
        img_path,label = self.samples[idx]
        try:
            #用PIL的Image打开图片
            img = Image.open(img_path).convert('RGB')
            #Bilinear双线性插值调整图片大小
            img = img.resize((self.img_size,self.img_size),Image.BILINEAR)

            #转换为numpy数组
            '''
            重点:一定要copy,防止内存出错
            '''
            img_array = np.array(img,dtype=np.float32).copy()

            #释放PIL对象
            img.close()
            del img

            #归一化到0-1之间
            img_array /= 255.0

            #标准化
            mean = np.array([0.485,0.456,0.406],dtype=  np.float32)
            std = np.array([0.229,0.224,0.225],dtype = np.float32)
            img_array = (img_array - mean) / std

            #调整维度顺序HWC->CHW
            #使用transpose函数，但是内存不连续了
            img_array = np.transpose(img_array, (2, 0, 1))

            #确保内存连续，防止jittor报错
            img_array = np.ascontiguousarray(img_array)

        except Exception as e:
            print(f"警告: 读取图片 {img_path} 失败: {e}")
            img_array = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
        
        return img_array,label
    def __len__(self):
        return len(self.samples)

def get_dataloader(root_dir,mode = 'train',
                batch_size = 32,
                img_size = 224,
                shuffle = True,
                num_workers = 0,
                sample_ratio = 0.1):
    
    dataset = TomatoDataset(root_dir,mode,img_size)

     # 如果采样比例小于1.0，则进行采样
    if sample_ratio < 1.0:
        original_len = len(dataset)
        sample_size = max(1, int(original_len * sample_ratio))  # 至少保留1个样本
        # 随机选择样本索引
        indices = np.random.choice(original_len, sample_size, replace=False)
        # 创建新的样本列表
        dataset.samples = [dataset.samples[i] for i in indices]
        print(f"{mode} 数据集: 从 {original_len} 个样本中采样了 {sample_size} 个样本 ({sample_ratio*100:.1f}%)")

    dataset.set_attrs(
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,###避免段错误
        drop_last = False
    )
    return dataset

if __name__ == '__main__':
    
    print("="*60)
    print("番茄疾病数据集测试")
    print("="*60)
    
    root = 'tomato_yolo_dataset'
    
    # 创建数据集
    print("\n创建训练数据集...")
    train_loader = get_dataloader(
        root_dir=root,
        mode='train',
        batch_size=4,  # 先用小batch测试
        img_size=224,
        num_workers=0  # 必须为0
    )
    
    # 测试数据加载
    print("\n测试批次加载:")
    try:
        for i, (images, labels) in enumerate(train_loader):
            print(f"Batch {i+1}: images={images.shape}, labels={labels.shape}")
            print(f"  Label values: {labels.data}")
            print(f"  Image range: [{float(images.min()):.3f}, {float(images.max()):.3f}]")
            
            if i >= 2:
                break
        
        print("\n✅ 测试成功!")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
     