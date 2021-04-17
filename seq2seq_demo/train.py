from dataset import train_data_loader
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import config as cfg
from tqdm import tqdm
import torch
import os

"""
训练流程
"""


def train(epoch):

    seq2seq = Seq2Seq().to(cfg.device)
    optimizer = Adam(seq2seq.parameters())

    if os.path.exists(cfg.model_save_path):
        seq2seq.load_state_dict(torch.load(cfg.model_save_path, map_location='cpu'))
        print('Load model params.')

    if os.path.exists(cfg.optimizer_save_path):
        seq2seq.load_state_dict(torch.load(cfg.model_save_path, map_location='cpu'))
        print('Load optimizer params.')

    for i in range(epoch):
        print(f"epoch: {i}")
        for index, (inputs, target, input_length, target_length) in enumerate(train_data_loader):
            inputs = inputs.to(cfg.device)
            target = target.to(cfg.device)

            decoder_outputs, _ = seq2seq(inputs, target, input_length, target_length)

            decoder_outputs = decoder_outputs.reshape(decoder_outputs.shape[0] * decoder_outputs.shape[1], -1)
            target = target.reshape(-1)

            loss = F.nll_loss(decoder_outputs, target, ignore_index=cfg.num_sequence.PAD)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % 100 == 0:
                print(f"epoch: {i} -> {index}/{len(train_data_loader)}, loss: {loss.item()}")

        torch.save(seq2seq.state_dict(), cfg.model_save_path)
        print('model save successfully.')
        torch.save(optimizer.state_dict(), cfg.optimizer_save_path)
        print('optimizer save successfully.')


if __name__ == '__main__':
    train(10)
