from transformers import PretrainedConfig
from typing import List

class DADAConfig(PretrainedConfig):
    model_type = "DADA"

    def __init__(
        self,
        seq_len: int = 100, 
        hidden_dim: int = 64, 
        d_model: int = 256, 
        bn_dims: List[int] = [16, 32, 64, 128, 192, 256], 
        k: int = 3,
        patch_len: int = 5, 
        mask_mode: str = "c", 
        depth: int = 10, 
        max_iters: int = 1e5,
        **kwargs,
    ):
        self.seq_len=seq_len, 
        self.hidden_dim=hidden_dim, 
        self.d_model=d_model, 
        self.bn_dims=bn_dims, 
        self.k=k,
        self.patch_len=patch_len, 
        self.mask_mode=mask_mode, 
        self.depth=depth, 
        self.max_iters=max_iters,
        super().__init__(**kwargs)