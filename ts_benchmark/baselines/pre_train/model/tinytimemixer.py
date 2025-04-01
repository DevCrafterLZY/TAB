import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/tsfm')

from torch import nn
import torch.nn.functional as F
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


class TinyTimeMixer(nn.Module):

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

        self.model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2", revision="512-720-r2"
        )
        

    def forward(self, inputs): 
        point_forecast = self.model(inputs).prediction_outputs
        return point_forecast[:, :self.context_length,:]
