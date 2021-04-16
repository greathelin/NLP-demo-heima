"""
测试用户词典
"""
import jieba
import config as cfg

jieba.load_userdict(cfg.user_dict_path)


def test_user_dict():
    sentence = '人工智能+python和c++哪个难'
    ret = jieba.lcut(sentence)
    print(ret)
