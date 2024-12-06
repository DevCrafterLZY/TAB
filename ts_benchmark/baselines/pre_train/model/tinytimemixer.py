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
        
        # self.no_training = True

        # Use main for 512-96 model
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM", revision="main"
        )
        if not config.use_p:
            for param in self.model.parameters():
                param.data.uniform_(-0.02, 0.02)
        

    def forward(self, inputs, x_mark_enc=None, x_mark_dec=None): 
        B, C, K = inputs.shape 
        if C != 512: # padding 0
            pad_total = 512 - C
            inputs = F.pad(inputs, (0, 0, pad_total, 0), mode='constant', value=0)
        point_forecast = self.model(inputs).prediction_outputs
        return point_forecast
