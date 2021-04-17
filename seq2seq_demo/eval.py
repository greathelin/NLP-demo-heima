from seq2seq import Seq2Seq
import config as cfg
import numpy as np
import torch

"""模型评估"""


def model_eval(size=10):
    # 1.测试数据
    data = [str(i) for i in np.random.randint(0, 1e8, size=[size])]
    data = sorted(data, key=lambda x: len(x), reverse=True)

    input_length = torch.LongTensor([len(i) for i in data])
    inputs = torch.LongTensor([cfg.num_sequence.transform(list(i), cfg.max_len) for i in data]).to(cfg.device)

    # 2.实例化模型，加载模型
    seq2seq = Seq2Seq().to(cfg.device)
    seq2seq.eval()
    seq2seq.load_state_dict(torch.load(cfg.model_save_path, map_location='cpu'))

    # 3.获取预测值
    indices = seq2seq.evaluate(inputs, input_length)
    indices = np.array(indices).transpose()

    # 4.反序列化观察结果
    result = []
    for line in indices:
        temp_result = cfg.num_sequence.inverse_transform(line)
        cur_line = ""
        for word in temp_result:
            if word == cfg.num_sequence.EOS_TAG:  # 训练时未使用结束符，该行代码无效
                break

            else:
                cur_line += word

        result.append(cur_line)

    print('data:', data)
    print('result:', result)


if __name__ == '__main__':
    model_eval(5)
