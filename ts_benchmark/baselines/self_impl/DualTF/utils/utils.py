import copy
from typing import Union, Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, val_loss2, model):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model):
        self.check_point = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

def split_before(
    data: Union[pd.DataFrame, np.ndarray], index: int
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
    """
    Split time series data into two parts at the specified index.

    :param data: Time series data to be segmented.
                 Can be a pandas DataFrame or a NumPy array.
    :param index: Split index position.
    :return: Tuple containing the first and second parts of the data.
    """
    if isinstance(data, pd.DataFrame):
        return data.iloc[:index, :], data.iloc[index:, :]
    elif isinstance(data, np.ndarray):
        return data[:index, :], data[index:, :]
    else:
        raise TypeError("Input data must be a pandas DataFrame or a NumPy array.")

def generate_rolling_samples(
        raw_data: np.ndarray,
        seq_len: int,
):
    samples = []
    data = raw_data
    offsets = np.sort(np.concatenate((np.arange(0, seq_len, 1),)))
    for i in range(data.shape[0] - seq_len + 1):
        samples.append(data[i + offsets, ...])
    samples = np.stack(samples, axis=0)
    return samples.astype(np.float64)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def _create_sequences(values, seq_length, stride, historical=False):
    seq = []
    if historical:
        for i in range(seq_length, len(values) + 1, stride):
            seq.append(values[i - seq_length:i])
    else:
        for i in range(0, len(values) - seq_length + 1, stride):
            seq.append(values[i: i + seq_length])

    return np.stack(seq)


def torch_fft_transform(seq):
    torch_seq = torch.from_numpy(seq)
    # freq length
    tp_cnt = seq.shape[1]
    tm_period = seq.shape[1]

    # FFT
    ft_ = torch.fft.fft(torch_seq, dim=1) / tm_period
    # Half
    ft_ = ft_[:, range(int(tm_period / 2)), :]
    # index
    val_ = np.arange(int(tp_cnt / 2))
    # freq axis
    freq = val_ / tm_period

    ffts_tensor = abs(ft_)
    ffts = ffts_tensor.numpy()
    return ffts, freq

def generate_frequency_grandwindow(x_trains, nest_length, step):
    grand_trains = []
    for grand in range(len(x_trains)):
        sub_x_trains = _create_sequences(x_trains[grand], nest_length, step)
        train_sequences, freq = torch_fft_transform(sub_x_trains)
        grand_trains.append(train_sequences)
    grand_train = np.array(grand_trains)
    grand_train_reshaped = grand_train.reshape(grand_train.shape[0],
                                               grand_train.shape[1] * grand_train.shape[2],
                                               grand_train.shape[3])

    return grand_train_reshaped

class TrainingLoader(Dataset):
    def __init__(self,x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index,:,:])
        return x

class GeneralLoader(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index,:,:])
        y = torch.FloatTensor(self.y[index,:,:])
        return x, y

def get_grand_fre_dataloader(x_trains, nest_length, step, batch_size):
    grand_train_reshaped = generate_frequency_grandwindow(x_trains, nest_length, step)
    train_loader = DataLoader(dataset=TrainingLoader(grand_train_reshaped),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    return train_loader