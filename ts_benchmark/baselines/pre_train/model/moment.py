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

        # self.no_training = True

        self.model = MOMENTPipeline.from_pretrained(
            "ts_benchmark/baselines/pre_train/checkpoints/Moment_base", 
            model_kwargs={
                'task_name': 'reconstruction',
                'mask_ratio': config.mask_ratio,
                # 'forecast_horizon': config.pred_len,
                # 'backcast_horizon': config.seq_len,
                # 'head_dropout': 0.1,
                # 'weight_decay': 0,
                # 'freeze_encoder': True, # Freeze the patch embedding layer
                # 'freeze_embedder': True, # Freeze the transformer encoder
                # 'freeze_head': False, # The linear forecasting head must be trained
            },
        )
        self.model.init()
        if not config.use_p:
            for param in self.model.parameters():
                param.data.uniform_(-0.02, 0.02)
        

    def forward(self, inputs, x_mark_enc=None, x_mark_dec=None):
        
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
