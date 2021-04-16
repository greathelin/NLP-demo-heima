import pandas as pd
from tqdm import tqdm
from lib import cut
import config as cfg
import json
import random

flags = [0, 0, 0, 0, 1]  # 1/5的数据作为测试集

# 闲聊的语料
xiaohuangji_path = r"D:\PycharmProjects\QA-BOT\corpus\classify\origin_corpus\xiaohuangji50w_nofenci.conv"

# 问答的语料
byhand_path = r"D:\PycharmProjects\QA-BOT\corpus\classify\origin_corpus\手动构造的问题.json"
crawled_path = r"D:\PycharmProjects\QA-BOT\corpus\classify\origin_corpus\爬虫爬取的问题.csv"


def keyword_in_line(line):
    """判断line中是否存在不符合要求的词"""
    keywords_list = ["c语言", "c++", "java", "项目管理", "人工智能", "python", "前端", "linux"]
    for word in line:
        if word in keywords_list:
            return True
        else:
            return False


def process_xiaohuangji(f_train, f_test, by_word=False):
    """处理小黄鸡语料"""
    num_train = 0
    num_test = 0
    flag = 0
    with open(xiaohuangji_path, encoding='utf-8') as fr:
        # TODO 句子长度为1，考虑删除

        for line in tqdm(fr.readlines()[:10000], desc="小黄鸡"):
            line = line.strip()

            if line.startswith("E"):
                flag = 0
                continue
            elif line.startswith("M"):
                if flag == 0:  # 第一个M出现
                    line = line[1:].strip()
                    flag = 1
                else:
                    continue  # 不需要第二个出现的M开头句子

            line_cuted = " ".join(cut(line.strip(), by_word=by_word))
            if not keyword_in_line(line_cuted):
                line_cuted = line_cuted.strip() + "\t" + "__label__chat"

                if random.choice(flags) == 0:
                    f_train.write(line_cuted + "\n")
                    num_train += 1
                else:
                    f_test.write(line_cuted + "\n")
                    num_test += 1

    return num_train, num_test


def process_byhand_data(f_train, f_test, by_word=False):
    """处理手动构造的数据"""
    num_train = 0
    num_test = 0
    with open(byhand_path, encoding='utf-8') as fr:
        total_lines = json.load(fr)
        for key in tqdm(total_lines, desc="byhand data"):
            for lines in total_lines[key]:
                for line in lines:
                    line_cuted = " ".join(cut(line.strip(), by_word=by_word))
                    if not keyword_in_line(line_cuted):
                        line_cuted = line_cuted.strip() + "\t" + "__label__QA"
                        if random.choice(flags) == 0:
                            f_train.write(line_cuted + "\n")
                            num_train += 1
                        else:
                            f_test.write(line_cuted + "\n")
                            num_test += 1

    return num_train, num_test


def process_crawled_data(f_train, f_test, by_word=False):
    """处理抓取的数据"""
    num_train = 0
    num_test = 0
    with open(crawled_path, encoding='utf-8') as fr:
        for line in tqdm(fr.readlines(), desc="crawled data"):
            line_cuted = " ".join(cut(line.strip(), by_word=by_word))
            if not keyword_in_line(line_cuted):
                line_cuted = line_cuted.strip() + "\t" + "__label__QA"
                if random.choice(flags) == 0:
                    f_train.write(line_cuted + "\n")
                    num_train += 1
                else:
                    f_test.write(line_cuted + "\n")
                    num_test += 1

    return num_train, num_test


def process(by_word=False):
    train_path = cfg.classify_corpus_train_path if not by_word else cfg.classify_corpus_by_word_train_path
    test_path = cfg.classify_corpus_test_path if not by_word else cfg.classify_corpus_by_word_test_path

    f_train = open(train_path, "w", encoding='utf-8')
    f_test = open(test_path, "w", encoding='utf-8')

    # 处理小黄鸡
    num_chat_train, num_chat_test = process_xiaohuangji(f_train, f_test, by_word)

    # 处理手动构造的句子
    num_qa_train, num_qa_test = process_byhand_data(f_train, f_test, by_word)

    # 处理抓取的句子
    _a, _b = process_crawled_data(f_train, f_test, by_word)
    num_qa_train += _a
    num_qa_test += _b

    print(f"训练集:{num_chat_train + num_qa_train}, 测试集:{num_chat_test + num_qa_test}")
    print(f"chat:{num_chat_train + num_chat_test}, QA语料: {num_qa_train + num_qa_test}")

    f_train.close()
    f_test.close()
