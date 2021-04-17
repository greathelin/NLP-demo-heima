import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg

"""
实现解码器
"""


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(len(cfg.num_sequence), cfg.embedding_dim, cfg.num_sequence.PAD)
        self.gru = nn.GRU(input_size=cfg.embedding_dim,
                          num_layers=cfg.num_layer,
                          hidden_size=cfg.hidden_size,
                          batch_first=True)
        self.fc = nn.Linear(cfg.hidden_size, len(cfg.num_sequence))

    def forward(self, target, encoder_hidden):
        # 1. 获取encoder的输出，作为decoder第一次的hidden_state(隐藏状态)
        decoder_hidden = encoder_hidden
        batch_size = target.shape[0]

        # 2. 准备decoder第一个时间步的输入，[batch_size, 1] SOS 作为输入
        decoder_input = (torch.ones([batch_size, 1], dtype=torch.int64) * cfg.num_sequence.SOS).to(cfg.device)

        # 3. 在第一个时间步上计算，得到第一个时间步的输出，hidden_state
        # 4. 把前一个时间步的输出进行计算，得到第一个最后的输出的结果
        # 5. 把前一次的hidden_state 作为当前时间步的hidden_state的输入，把前一次的输出，作为当前时间步的输入
        # 6. 循环4-5步骤

        # 保存预测的结果
        decoder_outputs = torch.zeros([batch_size, cfg.max_len + 2, len(cfg.num_sequence)]).to(cfg.device)

        for t in range(cfg.max_len + 2):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            # print('decoder_outputs:', decoder_outputs.shape)
            # 保存decoder_output_t到decoder_outputs中
            decoder_outputs[:, t] = decoder_output_t

            value, index = torch.topk(decoder_output_t, 1)
            decoder_input = index.to(cfg.device)

        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden):
        """
        计算每个时间上的结果
        :param decoder_input: [batch_size, 1]
        :param decoder_hidden: [1, batch_size, hidden_size]
        :return:
        """
        decoder_input_embedded = self.embedding(decoder_input)  # [batch_size, 1, embedding_dim]
        # print('decoder_input_embedded:', decoder_input_embedded.shape)

        # out: [batch_size, 1, hidden_size]
        # decoder_hidden: [1, batch_size, hidden_size]
        out, decoder_hidden = self.gru(decoder_input_embedded, decoder_hidden)

        out = out.squeeze(1)  # [batch_size, hidden_size]
        # print("out:", out.shape)
        output = F.log_softmax(self.fc(out), dim=-1)  # [batch_size, vocab_size]
        # print("output:", output.shape)

        return output, decoder_hidden

    def evaluate(self, encoder_hidden):
        """
        模型评估
        :param encoder_hidden:
        :return:
        """
        decoder_hidden = encoder_hidden  # [1, batch_size, hidden_size]
        batch_size = encoder_hidden.shape[1]
        decoder_input = (torch.ones([batch_size, 1], dtype=torch.int64) * cfg.num_sequence.SOS).to(cfg.device)

        indices = []
        # while True:
        for i in range(cfg.max_len + 5):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            value, index = torch.topk(decoder_output_t, 1)

            decoder_input = index.to(cfg.device)

            # if index == cfg.num_sequence.EOS:
            #     break

            indices.append(index.squeeze(-1).cpu().detach().numpy())

        return indices
