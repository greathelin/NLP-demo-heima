import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config as cfg

"""
准备数据集, Dataset, DataLoader
"""


class NumDataset(Dataset):
    def __init__(self):
        # 使用numpy随机创建一堆数字
        self.data = np.random.randint(1e8, 9e8, size=[500000])

    def __getitem__(self, index):
        data = list(str(self.data[index]))
        label = data + ["0"]
        return data, label

    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    ret = list(zip(*batch))
    return ret


train_data_loader = DataLoader(NumDataset(), batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collate_fn)

if __name__ == '__main__':
    for i in train_data_loader:
        print(i)
        break

