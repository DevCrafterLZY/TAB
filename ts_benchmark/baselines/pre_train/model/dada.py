import torch
from torch import nn
from einops import rearrange, repeat

import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/DADA')
# from ts_benchmark. models import Timer
from ts_benchmark.baselines.pre_train.submodules.Dada.DADA.modeling_DADA import DADA
from ts_benchmark.baselines.pre_train.submodules.Dada.DADA.configuration_DADA import DADAConfig

from transformers import AutoModel

class DadaModel(nn.Module):
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

        self.model = AutoModel.from_pretrained("/root/CATCH/ts_benchmark/baselines/pre_train/submodules/DADA/DADA", trust_remote_code=True)

     
    def forward(self, inputs):
        outputs = self.model.infer(inputs)
        forecast, _ = torch.median(outputs, dim=0)
        return forecast
