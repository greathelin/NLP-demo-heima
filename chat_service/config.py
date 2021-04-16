"""
配置文件
"""
# =================语料相关================
user_dict_path = "corpus/user_dict/user_dict.txt"
stopwords_path = "corpus/stopwords-master/cn_stopwords.txt"
classify_corpus_train_path = "corpus/classify/classify_train.txt"
classify_corpus_test_path = "corpus/classify/classify_test.txt"

classify_corpus_by_word_train_path = "corpus/classify/classify_train_by_word.txt"
classify_corpus_by_word_test_path = "corpus/classify/classify_test_by_word.txt"

# =================分类相关================
classify_model_path = "model/classify.model"  # 一个词语作为特征的模型 的保存地址
classify_model_path_by_word = "model/classify_by_word.model"  # 把单个字作为特征的模型 的保存地址

classify_model_final_path = "model/classify.model"  # 一个词语作为特征的模型
classify_model_final_path_by_word = "model/classify.model"  # 把单个字作为特征的模型
