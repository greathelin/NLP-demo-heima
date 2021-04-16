"""测试分类相关的API"""
from classify.build_model import build_classify_model, get_classify_model, eval_

if __name__ == '__main__':
    # build_classify_model(by_word=True)
    eval_acc = eval_(by_word=True)
    print(eval_acc)

    eval_acc = eval_(by_word=False)
    print(eval_acc)
