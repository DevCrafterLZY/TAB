import sys

import torch
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/moment')

from einops import rearrange
from torch import nn

from momentfm import MOMENTPipeline

class Moment(nn.Module):

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset

        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            # "ts_benchmark/baselines/pre_train/checkpoints/Moment-large",
            model_kwargs={"task_name": "reconstruction"},
        )
        self.model.init()
        

    def forward(self, inputs):
        
        B, L, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> b k l')
        # # 计算需要的填充大小
        padding_size = 8 - (L % 8)  # padding_size = 4
        padding_patch_layer = nn.ReplicationPad1d((0, padding_size))
        inputs = padding_patch_layer(inputs)
        point_forecast = self.model(x_enc=inputs)
        forecast = point_forecast.reconstruction[:, :, :self.context_length]
        output = rearrange(forecast, 'b k l -> b l k', b=B, k=K)
        return output
