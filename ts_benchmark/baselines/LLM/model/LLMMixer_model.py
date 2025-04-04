import torch
from torch import nn

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/submodules/LLMMixer")

from ts_benchmark.baselines.LLM.submodules.LLMMixer.models import LLMMixer

class LLMMixerModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        # config.pred_len = config.horizon
        self.model = LLMMixer.Model(config)
       
    def forward(self, x_enc):        
        output = self.model.anomaly_detection(x_enc)
        return output
