from fasttext import FastText
import config as cfg


def build_classify_model(by_word=False):
    """
    :param by_word:是否使用单个字作为特征
    :return:
    """
    data_path = cfg.classify_corpus_train_path if not by_word else cfg.classify_corpus_by_word_train_path
    model = FastText.train_supervised(data_path, epoch=20, wordNgrams=1, minCount=5)
    save_path = cfg.classify_model_path if not by_word else cfg.classify_model_path_by_word
    model.save_model(save_path)


def get_classify_model(by_word=False):
    """加载model"""
    save_path = cfg.classify_model_path if not by_word else cfg.classify_model_path_by_word
    model = FastText.load_model(save_path)
    return model


def eval_(by_word=False):
    """
    模型的评估，获取模型的准确率
    :param by_word:是否使用单个字作为特征
    :return:
    """
    model = get_classify_model(by_word)
    inputs = []
    target = []
    eval_data_path = cfg.classify_corpus_test_path if not by_word else cfg.classify_corpus_by_word_test_path
    for line in open(eval_data_path, encoding="utf-8").readlines():
        temp = line.strip().split("__label__")
        if len(temp) < 2:
            continue
        inputs.append(temp[0].strip())
        target.append(temp[1].strip())

    # 使用特征和模型进行预测
    labels, acc_list = model.predict(inputs)

    # 计算准确率
    sum_ = 0
    print(len(labels), len(target))
    for i, j in zip(labels, target):
        if i[0].replace("__label__", "").strip() == j:
            sum_ += 1
    acc = sum_ / len(labels)  # 平均的准确率

    return acc
