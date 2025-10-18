# %%
import jittor as jt
from jittor import nn
import math

import sys
import os

current_dir = os.path.abspath('.')
project_root = current_dir

for _ in range(2):
    project_root = os.path.dirname(project_root)
print(f"Project root determined as: {project_root}")

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)
        

# %%
from config import Config
config = Config()

# %%
class patch_embedding_Layer(nn.Module):
    def __init__(self, 
                img_size=config.IMG_SIZE, 
                in_channels=config.IN_CHANNELS, 
                patch_size=config.PATCH_SIZE, 
                embed_dim=config.EMBED_DIM):
        super(patch_embedding_Layer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 使用可学习的参数
        self.class_token = jt.init.gauss((1, 1, embed_dim))
        self.position_embed = jt.init.gauss((1, 1 + self.num_patches, embed_dim))
        
        # 添加归一化层
        self.norm = nn.LayerNorm(embed_dim)

    def execute(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size,\
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        x = self.proj(x).view(B, self.num_patches, -1)
        
        # 添加class token
        x = jt.concat([self.class_token.expand(B, -1, -1), x], dim=1)
        
        # 添加位置编码
        x = x + self.position_embed.expand(B, -1, -1)
        
        # 添加归一化
        x = self.norm(x)
        
        return x


# %%



