import torch
from torch import nn
from einops import rearrange, repeat

import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/Timer')
# from ts_benchmark. models import Timer
from ts_benchmark.baselines.pre_train.submodules.Timer.models import Timer


class TimerModel(nn.Module):
# class Timer(nn.Module):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset
        self.config = config

        self.output_patch_len = 96 # fixed by the pre-trained model
        self.label_len = config.seq_len - 96

        class TimerConfig:
            def __init__(self):
                self.task_name = 'anomaly_detection'
                self.ckpt_path = 'ts_benchmark/baselines/pre_train/checkpoints/timer/Timer_anomaly_detection_1.0.ckpt'
                self.patch_len =  96
                self.d_model = 256
                self.d_ff = 512
                self.e_layers = 4
                self.n_heads = 8
                self.dropout = 0.1
                self.output_attention = False
                self.factor = 1
                self.activation = 'gelu'

        model_config = TimerConfig()
        self.model = Timer.Model(model_config)
        if not config.use_p:
            for param in self.timer.parameters():
                param.data.uniform_(-0.02, 0.02)
       
     
    def forward(self, inputs):
            
        B, L, K = inputs.shape

        inputs = rearrange(inputs, 'b l k -> b k l')

        inputs = rearrange(inputs, 'b k l -> (b k) l 1')

        outputs = self.model.anomaly_detection(inputs)
        pred_y = rearrange(outputs, '(b k) l 1 -> b l k', b=B, k=K)
            
        return pred_y[:, :self.context_length, :]
