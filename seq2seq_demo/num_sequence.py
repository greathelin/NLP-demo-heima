class NumSequence:
    PAD_TAG = "PAD"
    UNK_TAG = "UNK"
    SOS_TAG = "SOS"  # start of sequence 句子开始的符号
    EOS_TAG = "EOS"  # 结束符
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {self.PAD_TAG: self.PAD,
                     self.UNK_TAG: self.UNK,
                     self.SOS_TAG: self.SOS,
                     self.EOS_TAG: self.EOS,
                     }
        for i in range(10):
            self.dict[str(i)] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len, add_eos=False):
        """
        把sentence转化为数字序列

        add_eos: True, 输出句子长度为 max_len + 1
                False, 输出句子长度为 max_len
        """
        if len(sentence) > max_len:  # 句子的长度和max_len一样长的时候
            sentence = sentence[:max_len]

        sentence_len = len(sentence)  # 提前计算句子长度，实现add_eos后，句子长度统一

        if add_eos:
            sentence = sentence + [self.EOS_TAG]

        if sentence_len < max_len:
            sentence = sentence + [self.PAD_TAG] * (max_len - sentence_len)  # 填充

        result = [self.dict.get(i, self.UNK) for i in sentence]

        return result

    def inverse_transform(self, indices):
        """把序列转回字符串"""
        return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    n = NumSequence()
    print(n.dict)
