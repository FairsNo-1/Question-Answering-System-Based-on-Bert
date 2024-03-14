# Edited by ZhangWen Lu
# Date:2024.1.2
# Env: tensorflow 2.12.0
#      pytorch 1.12.1

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import torch


def get_answer_using_bert(question, reference_text):

    """
    # 这里是采用在线下载的方式从清华镜像源下载预训练模型，如果网络状况较好可以采用
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',
                                                    mirror='tuna')
    bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',
                                                           mirror='tuna')
    """

    # 加载问答预训练模型bert-large-uncased-whole-word-masking-finetuned-squad（包括模型model和分词器tokenizer)
    # 请在改变运行环境时正确修改模型路径
    pretrained_path = 'D:/桌面/Bert_Q&A/bert-large-uncased-whole-word-masking-finetuned-squad'
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    bert_model = BertForQuestionAnswering.from_pretrained(pretrained_path)

    # 对输入文本和问题进行分词
    input_ids = bert_tokenizer.encode(question, reference_text)
    input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

    # 查找第一次出现的[SEP]令牌的索引（[SEP]表示分割符，[CLS]表示起始标识）
    sep_location = input_ids.index(bert_tokenizer.sep_token_id)
    first_seg_len, second_seg_len = sep_location + 1, len(input_ids) - (sep_location + 1)
    seg_embedding = [0] * first_seg_len + [1] * second_seg_len

    # 测试模型
    model_scores = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([seg_embedding]))
    ans_start_loc, ans_end_loc = torch.argmax(model_scores[0]), torch.argmax(model_scores[1])
    result = ' '.join(input_tokens[ans_start_loc:ans_end_loc + 1])
    result = result.replace(' ##', '')
    if result[0] == '[CLS]':
        # 回答失败反馈
        result = 'unable to find the answer'
    return result


def question_answer():
    # 加载CoQA数据集的CSV文件
    data = pd.read_csv('CoQA_data.csv')

    # 从CoQA数据集中随机选取问题和原文进行答案预测并打印结果
    '''
    # 测试用问题
    question = "Where was the Football League founded?"
    reference_text = " In 1888, The Football League was founded in England, becoming the first of many professional foot
                        ball competitions. During the 20th century, several of the various kinds of football grew to 
                        become some of the most popular team sports in the world."
    '''
    while True:
        random_num = np.random.randint(0, len(data))
        reference_text = data["text"][random_num]
        question = data["question"][random_num]
        real_answer = data["answer"][random_num]
        print(f"context: {reference_text}")
        print(f"question: {question}")
        print(f"predicted answer: {get_answer_using_bert(question, reference_text)}")
        print(f"real answer: {real_answer}")
        # 选择是否继续测试
        print("keep asking?(y for yes,n for no)")
        response = input()
        if response == 'n':
            break
        # print("\n")


if __name__ == "__main__":
    question_answer()
