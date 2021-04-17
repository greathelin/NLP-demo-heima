import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config as cfg
import torch

"""
准备数据集, Dataset, DataLoader
"""


class NumDataset(Dataset):
    def __init__(self):
        # 使用numpy随机创建一堆数字
        # np.random.seed(10)
        self.data = np.random.randint(0, 1e8, size=[500000])

    def __getitem__(self, index):
        inputs = list(str(self.data[index]))
        target = inputs + ["0"]
        inputs_length = len(inputs)
        target_length = len(target)
        return inputs, target, inputs_length, target_length

    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    """
    :param batch: [(inputs, target, inputs_length, target_length), (inputs, target, inputs_length, target_length)...]
    :return:
    """
    batch = sorted(batch, key=lambda x: x[3], reverse=True)  # 降序排序
    inputs, target, inputs_length, target_length = zip(*batch)
    # 把inputs转为序列
    inputs = torch.LongTensor([cfg.num_sequence.transform(i, max_len=cfg.max_len) for i in inputs])
    target = torch.LongTensor([cfg.num_sequence.transform(i, max_len=cfg.max_len + 2) for i in target])
    inputs_length = torch.LongTensor(inputs_length)
    target_length = torch.LongTensor(target_length)
    return inputs, target, inputs_length, target_length


train_data_loader = DataLoader(NumDataset(), batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collate_fn)
