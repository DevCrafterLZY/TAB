'''
* @author: EmpyreanMoon
*
* @create: 2024-08-25 20:20
*
* @description: various forms of frequency loss
'''

import torch
from einops import rearrange
import numpy as np


class frequency_loss(torch.nn.Module):
    def __init__(self, configs, keep_dim=False, dim=None):
        super(frequency_loss, self).__init__()
        self.keep_dim = keep_dim
        self.dim = dim
        if configs.auxi_mode == "fft":
            self.fft = torch.fft.fft
        elif configs.auxi_mode == "rfft":
            self.fft = torch.fft.rfft
        else:
            raise NotImplementedError
        self.configs = configs
        if configs.mask:
            self._generate_mask()
        else:
            self.mask = None

    def _generate_mask(self):
        if self.configs.add_noise and self.configs.noise_amp > 0:
            seq_len = self.configs.pred_len
            cutoff_freq_percentage = self.configs.noise_freq_percentage
            if self.configs.auxi_mode == "rfft":
                cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            elif self.configs.auxi_mode == "fft":
                cutoff_freq = int((seq_len) * cutoff_freq_percentage)
                low_pass_mask = torch.ones(seq_len)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1)
        else:
            self.mask = None

    def forward(self, outputs, batch_y):
        if outputs.is_complex():
            frequency_outputs = outputs
        else:
            frequency_outputs = self.fft(outputs, dim=1)
        # fft shape: [B, P, D]
        if self.configs.auxi_type == 'complex':
            loss_auxi = frequency_outputs - self.fft(batch_y, dim=1)
        elif self.configs.auxi_type == 'complex-phase':
            loss_auxi = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
        elif self.configs.auxi_type == 'complex-mag-phase':
            loss_auxi_mag = (frequency_outputs - self.fft(batch_y, dim=1)).abs()
            loss_auxi_phase = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        elif self.configs.auxi_type == 'phase':
            loss_auxi = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
        elif self.configs.auxi_type == 'mag':
            loss_auxi = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
        elif self.configs.auxi_type == 'mag-phase':
            loss_auxi_mag = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
            loss_auxi_phase = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        else:
            raise NotImplementedError

        if self.mask is not None:
            loss_auxi *= self.mask

        if self.configs.auxi_loss == "MAE":
            loss_auxi = loss_auxi.abs().mean(dim=self.dim,
                                             keepdim=self.keep_dim) if self.configs.module_first else loss_auxi.mean(
                dim=self.dim, keepdim=self.keep_dim).abs()  # check the dim of fft
        elif self.configs.auxi_loss == "MSE":
            loss_auxi = (loss_auxi.abs() ** 2).mean(dim=self.dim,
                                                    keepdim=self.keep_dim) if self.configs.module_first else (
                    loss_auxi ** 2).mean(dim=self.dim, keepdim=self.keep_dim).abs()
        else:
            raise NotImplementedError
        return loss_auxi


class frequency_criterion(torch.nn.Module):
    def __init__(self, configs):
        super(frequency_criterion, self).__init__()
        self.metric = frequency_loss(configs, dim=1, keep_dim=True)
        self.patch_size = configs.inference_patch_size
        self.patch_stride = configs.inference_patch_stride
        self.win_size = configs.seq_len
        self.patch_num = int((self.win_size - self.patch_size) / self.patch_stride + 1)
        self.padding_length = self.win_size - (self.patch_size + (self.patch_num - 1) * self.patch_stride)

    def forward(self, outputs, batch_y):

        output_patch = outputs.unfold(dimension=1, size=self.patch_size,
                                      step=self.patch_stride)

        b, n, c, p = output_patch.shape
        output_patch = rearrange(output_patch, 'b n c p -> (b n) p c')
        y_patch = batch_y.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        y_patch = rearrange(y_patch, 'b n c p -> (b n) p c')

        main_part_loss = self.metric(output_patch, y_patch)
        main_part_loss = main_part_loss.repeat(1, self.patch_size, 1)
        main_part_loss = rearrange(main_part_loss, '(b n) p c -> b n p c', b=b)

        end_point = self.patch_size + (self.patch_num - 1) * self.patch_stride - 1
        start_indices = np.array(range(0, end_point, self.patch_stride))
        end_indices = start_indices + self.patch_size

        indices = torch.tensor([range(start_indices[i], end_indices[i]) for i in range(n)]).unsqueeze(0).unsqueeze(-1)
        indices = indices.repeat(b, 1, 1, c).to(main_part_loss.device)
        main_loss = torch.zeros((b, n, self.win_size - self.padding_length, c)).to(main_part_loss.device)
        main_loss.scatter_(dim=2, index=indices, src=main_part_loss)

        non_zero_cnt = torch.count_nonzero(main_loss, dim=1)
        main_loss = main_loss.sum(1) / non_zero_cnt

        if self.padding_length > 0:
            padding_loss = self.metric(outputs[:, -self.padding_length:, :], batch_y[:, -self.padding_length:, :])
            padding_loss = padding_loss.repeat(1, self.padding_length, 1)
            total_loss = torch.concat([main_loss, padding_loss], dim=1)
        else:
            total_loss = main_loss
        return total_loss
