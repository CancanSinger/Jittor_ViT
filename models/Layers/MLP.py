# %%
import jittor as jt
from jittor import nn

import sys
import os

current_dir = os.path.abspath('.')
project_root = current_dir

for _ in range(2):
    project_root = os.path.dirname(project_root)

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)
from config import Config
config = Config()


# %%
class MLP_Layer(nn.Module):
    def __init__(self, embedded_dim,hidden_dim,out_dim,dropout_rate=config.DROPOUT):
        super(MLP_Layer,self).__init__()
        self.fc1 = nn.Linear(embedded_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def execute(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


