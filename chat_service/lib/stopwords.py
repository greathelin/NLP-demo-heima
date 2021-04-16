"""
获取停用词
"""
import config as cfg


def stopwords():
    with open(cfg.stopwords_path, encoding='utf-8') as f:
        return [i.strip() for i in f.readlines()]
