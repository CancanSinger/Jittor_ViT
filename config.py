#config.py
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
        self.MLP_Hidden_Dim = 768*4    # MLP隐藏层维度
        self.MLP_RATIO = 4           # MLP扩展比例
        self.NUM_CLASSES = 80      # 分类任务的类别数