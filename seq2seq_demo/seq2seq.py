import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
from encoder import Encoder
from decoder import Decoder

"""
把encoder和decoder进行合并，得到seq2seq模型
"""


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inputs, target, input_length, target_length):
        encoder_outputs, encoder_hidden = self.encoder(inputs, input_length)
        decoder_outputs, decoder_hidden = self.decoder(target, encoder_hidden)
        return decoder_outputs, decoder_hidden

    def evaluate(self, inputs, input_length):
        encoder_outputs, encoder_hidden = self.encoder(inputs, input_length)
        indices = self.decoder.evaluate(encoder_hidden)
        return indices
