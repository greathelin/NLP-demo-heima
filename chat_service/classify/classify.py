import config as cfg
from fasttext import FastText

"""
意图识别模型的封装
"""


class Classify:
    def __init__(self, by_word=False):
        """
        加载训练好的模型
        """
        # TODO 从两个模型中选择最好最好的结果返回
        if by_word:
            self.model = FastText.load_model(cfg.classify_model_final_path)
        else:
            self.model = FastText.load_model(cfg.classify_model_final_path_by_word)

    def predict(self, sentence_cut):
        """
        预测输入数据的结果，准确率
        :param sentence_cut: 分词之后的句子
        :return: (label, accuracy)
        """
        result = self.model.predict(sentence_cut)
        for label, acc in zip(*result):
            # 把所有的label和acc转化到label_chat上比较其准确率
            if label == "__label__chat":
                label = "__label__QA"
                acc = 1 - acc

            # 判断准确率
            if acc > 0.95:
                return ("QA", acc)
            else:
                return ("chat", 1 - acc)

            # TODO 假设有三个类别
