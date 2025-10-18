# %%
import jittor as jt
from jittor import nn

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
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        
    def execute(self, x: jt.Var) -> jt.Var:
        return self.norm(x)



