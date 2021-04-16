from prepare_corpus.prepare_user_dict.test_user_dict import test_user_dict
from lib.cut_sentence import cut
from lib import stopwords

if __name__ == '__main__':
    # test_user_dict()
    s = "python难不难，是不是很难，有一些难"
    print(cut(s, with_sg=False, use_stopwords=True))
    # print(stopwords)
