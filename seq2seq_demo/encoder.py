import torch.nn as nn
import config as cfg
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

"""
编码器
"""


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(cfg.num_sequence),
                                      embedding_dim=cfg.embedding_dim,
                                      padding_idx=cfg.num_sequence.PAD)
        self.gru = nn.GRU(input_size=cfg.embedding_dim,
                          num_layers=cfg.num_layer,
                          hidden_size=cfg.hidden_size,
                          batch_first=True)

    def forward(self, inputs, input_length):
        """
        :param inputs: [batch_size, max_len]
        :param input_length:
        :return:
        """
        embedded = self.embedding(inputs)  # [batch_size, max_len, embedding_dim]

        embedded = pack_padded_sequence(embedded, input_length, batch_first=True)
        out, hidden = self.gru(embedded)

        # 解包
        out, out_length = pad_packed_sequence(out, batch_first=True, padding_value=cfg.num_sequence.PAD)
        return out, hidden
