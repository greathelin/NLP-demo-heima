"""
分词
"""
import jieba
import jieba.posseg as psg
import string
import config as cfg
from lib import stopwords
import logging

# 关闭jieba log输出
jieba.setLogLevel(logging.INFO)

jieba.load_userdict(cfg.user_dict_path)
# 准备英文字符
letters = string.ascii_lowercase + "+"


def cut_sentence_by_word(sentence):
    """
    实现中英文分词
    :param sentence:
    :return:
    """
    # python和c++哪个难？--> [python, 和, c++, 哪, 个, 难, ？]
    result = []
    temp = ""
    for word in sentence:
        # 把英文单词进行拼接
        if word.lower() in letters:
            temp += word
        else:
            if temp != "":  # 出现中文，把英文添加到结果中
                result.append(temp.lower())
                temp = ""

            if len(word.strip()) > 0:
                result.append(word.strip())

    if temp != "":  # 判断最后的字符是否为英文
        result.append(temp.lower())

    return result


def cut(sentence: str, by_word=False, use_stopwords=False, with_sg=False):
    """
    :param sentence: str 句子
    :param by_word: 是否按照单个字分词
    :param use_stopwords: 是否使用停用词
    :param with_sg: 是否返回词性
    :return:
    """
    sentence = sentence.strip()

    if by_word:
        result = cut_sentence_by_word(sentence)
    else:
        result = psg.lcut(sentence)
        result = [(i.word, i.flag) for i in result if len(i.word.strip()) > 0]
        if not with_sg:
            result = [i[0] for i in result if len(i[0].strip()) > 0]

    # 是否使用停用词
    if use_stopwords:
        result = [i for i in result if i not in stopwords()]

    return result


if __name__ == '__main__':
    r = cut('python和c++哪个难？ 嘿嘿UI/UE呢haha', False, False, False)
    print(r)
