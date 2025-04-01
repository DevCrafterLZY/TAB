from typing import Optional, Callable

import torch
from torch import nn

from ts_benchmark.baselines.self_impl.TFAD.model.distances import LpDistance
from ts_benchmark.baselines.self_impl.TFAD import model


def D_matrix(N):
    D = torch.zeros(N - 1, N)
    D[:, 1:] = torch.eye(N - 1)
    D[:, :-1] -= torch.eye(N - 1)
    return D


class hp_filter(nn.Module):
    """
        Hodrick Prescott Filter to decompose the series
    """

    def __init__(self, lamb):
        super(hp_filter, self).__init__()
        self.lamb = lamb

    def forward(self, x):
        x = x.permute(0, 2, 1)
        N = x.shape[1]
        D1 = D_matrix(N)
        D2 = D_matrix(N - 1)
        D = torch.mm(D2, D1).to(device='cuda')

        g = torch.matmul(torch.inverse(torch.eye(N).to(device='cuda') + self.lamb * torch.mm(D.T, D)), x)
        res = x - g
        g = g.permute(0, 2, 1)
        res = res.permute(0, 2, 1)
        return res, g


class TFADModel(nn.Module):
    """Neural Contrastive Detection in Time Series"""

    def __init__(self, configs):
        super(TFADModel, self).__init__()

        self.learning_rate = configs.lr
        self.k = configs.k
        self.time = 0
        self.classification_loss = nn.BCELoss(),
        self.config = configs

        # Encoder Network
        self.encoder1 = model.TCNEncoder(
            in_channels=self.config.feats,
            out_channels=self.config.embedding_rep_dim,
            kernel_size=self.config.tcn_kernel_size,
            tcn_channels=self.config.tcn_out_channels,
            tcn_layers=self.config.tcn_layers,
            tcn_out_channels=self.config.tcn_out_channels,
            maxpool_out_channels=self.config.tcn_maxpool_out_channels,
            normalize_embedding=self.config.normalize_embedding,
        )

        self.encoder2 = model.TCNEncoder(
            in_channels=self.config.feats,
            out_channels=self.config.embedding_rep_dim,
            kernel_size=self.config.tcn_kernel_size,
            tcn_channels=self.config.tcn_out_channels,
            tcn_layers=self.config.tcn_layers,
            tcn_out_channels=self.config.tcn_out_channels,
            maxpool_out_channels=self.config.tcn_maxpool_out_channels,
            normalize_embedding=self.config.normalize_embedding,
        )

        self.encoder1f = model.TCNEncoder(
            in_channels=self.config.feats,
            out_channels=self.config.embedding_rep_dim,
            kernel_size=self.config.tcn_kernel_size,
            tcn_channels=self.config.tcn_out_channels,
            tcn_layers=self.config.tcn_layers,
            tcn_out_channels=self.config.tcn_out_channels,
            maxpool_out_channels=self.config.tcn_maxpool_out_channels,
            normalize_embedding=self.config.normalize_embedding,
        )

        self.encoder2f = model.TCNEncoder(
            in_channels=self.config.feats,
            out_channels=self.config.embedding_rep_dim,
            kernel_size=self.config.tcn_kernel_size,
            tcn_channels=self.config.tcn_out_channels,
            tcn_layers=self.config.tcn_layers,
            tcn_out_channels=self.config.tcn_out_channels,
            maxpool_out_channels=self.config.tcn_maxpool_out_channels,
            normalize_embedding=self.config.normalize_embedding,
        )
        self.Decomp1 = hp_filter(lamb=self.config.hp_lamb)
        # Contrast Classifier
        self.classifier = model.ContrastiveClasifier(
            distance=LpDistance(),
        )

    def forward(self, x):
        # The encoder could manage other window lengths,
        # but all training and validation is currently performed with a single length
        assert x.shape[-1] == self.config.window_length

        res, cyc = self.Decomp1(x)

        ts_whole_res_emb = self.encoder1(res)
        ts_context_res_emb = self.encoder1(res[..., : -self.config.suspect_window_length])

        ts_whole_cyc_emb = self.encoder2(cyc)
        ts_context_cyc_emb = self.encoder2(cyc[..., : -self.config.suspect_window_length])

        res_fft_whole = torch.fft.fft(res, dim=-1, norm='forward')
        cyc_fft_whole = torch.fft.fft(cyc, dim=-1, norm='forward')
        res_temp_whole = torch.cat((res_fft_whole.real, res_fft_whole.imag), -3)
        res_fft_ric_whole = torch.reshape(res_temp_whole.permute(1, 2, 0),
                                          [res_fft_whole.shape[-3], res_fft_whole.shape[-2], -1])
        cyc_temp_whole = torch.cat((cyc_fft_whole.real, cyc_fft_whole.imag), -3)
        cyc_fft_ric_whole = torch.reshape(cyc_temp_whole.permute(1, 2, 0),
                                          [cyc_fft_whole.shape[-3], cyc_fft_whole.shape[-2], -1])

        res_con = res[..., : -self.config.suspect_window_length]
        cyc_con = cyc[..., : -self.config.suspect_window_length]

        res_fft_con = torch.fft.fft(res_con, dim=-1, norm='forward')
        cyc_fft_con = torch.fft.fft(cyc_con, dim=-1, norm='forward')
        res_temp_con = torch.cat((res_fft_con.real, res_fft_con.imag), -3)
        res_fft_ric_con = torch.reshape(res_temp_con.permute(1, 2, 0),
                                        [res_fft_con.shape[-3], res_fft_con.shape[-2], -1])
        cyc_temp_con = torch.cat((cyc_fft_con.real, cyc_fft_con.imag), -3)
        cyc_fft_ric_con = torch.reshape(cyc_temp_con.permute(1, 2, 0),
                                        [cyc_fft_con.shape[-3], cyc_fft_con.shape[-2], -1])

        ts_whole_res_emb_f = self.encoder1f(res_fft_ric_whole)
        ts_context_res_emb_f = self.encoder1f(res_fft_ric_con)

        ts_whole_cyc_emb_f = self.encoder2f(cyc_fft_ric_whole)
        ts_context_cyc_emb_f = self.encoder2f(cyc_fft_ric_con)

        logits_anomaly = self.classifier(ts_whole_res_emb, ts_context_res_emb, ts_whole_res_emb_f, ts_context_res_emb_f,
                                         ts_whole_cyc_emb, ts_context_cyc_emb, ts_whole_cyc_emb_f, ts_context_cyc_emb_f,
                                         self.config.weight_fft_branch)

        return logits_anomaly
