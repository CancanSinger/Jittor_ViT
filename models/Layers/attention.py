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


if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)
        

# %%
from config import Config
config = Config()

# %%
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embedded_dim,num_heads,dropout_rate=config.DROPOUT):
        super(MultiHeadSelfAttention,self).__init__()
        self.embedded_dim = embedded_dim
        self.num_heads = num_heads
        self.every_head_dim = embedded_dim // num_heads
        self.scale = math.sqrt(self.every_head_dim)
        self.dropout_rate = dropout_rate

        self.qkv_layer = nn.Linear(embedded_dim,embedded_dim*3)
        self.out_layer = nn.Linear(embedded_dim,embedded_dim)

        self.attn_drop = nn.Dropout(dropout_rate)


    def execute(self,x:jt.Var,mask:jt.Var=None)->jt.Var:
        B,N,C = x.shape
        # [B,N,embedded_dim]->[B,N,3*embedded_dim]
        qkv = self.qkv_layer(x)  
        # [B,N,3*embedded_dim]->[B,N,3,num_heads,every_head_dim]
        qkv = qkv.view(B,N,3,self.num_heads,self.every_head_dim)
        #[B,N,3,num_heads,every_head_dim]-> 3* [B,N,num_heads,every_head_dim]
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        #[B,N,num_heads,every_head_dim]->[B,num_heads,N,every_head_dim]
        q=q.permute(0,2,1,3)
        k=k.permute(0,2,1,3)
        v=v.permute(0,2,1,3)

        #计算注意力得分，得分维度为[B, num_heads, N, N]
        attention_scores = jt.matmul(q,k.transpose(0,1,3,2))/self.scale
        #应用mask（如果有的话）
        if mask is not None:
            mask = mask.view(mask.shape[0], 1, 1, mask.shape[1])
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        #应用softmax
        attention_scores = nn.softmax(attention_scores, dim=-1)
        attention_scores = self.attn_drop(attention_scores)
        #计算注意力加权值
        # [B,num_heads,N,N]*[B,num_heads,N,every_head_dim]->[B,num_heads,N,every_head_dim]
        attention_output = jt.matmul(attention_scores, v)
        # [B,num_heads,N,every_head_dim]->[B,N,embedded_dim]
        attention_output = attention_output.permute(0,2,1,3).reshape(B,N,C)
        #通过输出线性层
        output = self.out_layer(attention_output)
        #out的维度为[B,N,embedded_dim]
        return output   



