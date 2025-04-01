import torch
from torch import nn

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/submodules/CALF")

from ts_benchmark.baselines.LLM.submodules.CALF.models import CALF

class CALFModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        # config.pred_len = config.horizon
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CALF.Model(config, device)
       
    def forward(self, x_enc):        
        output = self.model(x_enc)
        return output['outputs_time']
