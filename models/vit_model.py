
import sys
import os

current_dir = os.path.abspath('.')
project_root = current_dir
project_root = os.path.dirname(project_root)

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)

import jittor as jt
from jittor import nn
import math
from config import Config
config = Config()

from models.Layers.layer_norm import LayerNorm
from models.Layers.MLP import MLP_Layer
from models.Layers.attention import MultiHeadSelfAttention
from models.Layers.patch_embed import patch_embedding_Layer

class Block(nn.Module):
    def __init__(self,
                embed_dim=config.EMBED_DIM,
                num_heads=config.NUM_HEADS,
                dropout_rate=config.DROPOUT,
                hidden_dim=config.MLP_Hidden_Dim):
        super(Block, self).__init__()

        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(embed_dim)
        self.Multi_Head_Self_Attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.MLP = MLP_Layer(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout_rate=dropout_rate
        )
    def execute(self, x):
        y = self.norm1(x)
        y = self.Multi_Head_Self_Attention(y)
        x = x + y

        z = self.norm2(x)
        z = self.MLP(z)
        x = x + z

        return x

class Visual_Transformer(nn.Module):
    def __init__(self,
                img_size=config.IMG_SIZE,
                patch_size=config.PATCH_SIZE,
                in_channels=config.IN_CHANNELS,

                embed_dim=config.EMBED_DIM,
                depth=config.NUM_LAYERS,
                num_heads=config.NUM_HEADS,
                dropout_rate=config.DROPOUT,
                hidden_dim=config.MLP_Hidden_Dim):
        super(Visual_Transformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        self.patch_embed = patch_embedding_Layer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )   
        self.blocks = nn.ModuleList([
            Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                hidden_dim=hidden_dim
            ) for _ in range(depth)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, config.NUM_CLASSES)
        )

    def execute(self, x):
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        class_token = x[:, 0]
        logits = self.classifier(class_token)

        return logits
