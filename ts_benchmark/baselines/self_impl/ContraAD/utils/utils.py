import torch
import torch.nn.functional as F

from einops import repeat


def normalize(x, method='softmax'):
    # x shape : batch, size, channel
    # min-max normalization
    if len(x.shape) < 3:
        x = x.unsqueeze(dim=-1)
    b, w, c = x.shape
    min_vals, _ = torch.min(x, dim=1)  # batch,channel
    max_vals, _ = torch.max(x, dim=1)  # batch,channel
    mean_vals = torch.mean(x, dim=1)
    std_vals = torch.std(x, dim=1)
    if method == 'min-max':
        min_vals = repeat(min_vals, 'b c -> b w c', w=w)
        max_vals = repeat(max_vals, 'b c -> b w c', w=w)
        x = (x - min_vals) / (max_vals - min_vals + 1e-8)
    # z-score normalization
    elif method == 'z-score':
        mean_vals = repeat(mean_vals, 'b c -> b w c', w=w)
        std_vals = repeat(std_vals, 'b c -> b w c', w=w)
        x = torch.abs((x - mean_vals) / (std_vals + 1e-8))
    # softmax normalization
    elif method == 'softmax':
        x = F.softmax(x, dim=1)
    else:
        raise ValueError('Unknown normalization method')
    if c == 1:
        x = x.squeeze(dim=-1)
    return x
