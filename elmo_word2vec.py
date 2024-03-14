# Edited by ZhangWen Lu
# Date:2024.1.1
# Env: tensorflow 2.12.0
#      pytorch 1.12.1

import tensorflow as tf
import tensorflow_hub as hub


if __name__ == '__main__':
    '''
    # 使用tensorflow_hub加载elmo模型进行词向量预训练
    # 从kaggle下载elmo的预训练模型，需要VPN，慎用
    elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True)
    '''
    # 禁用 Eager Execution(不禁用会导致tensorflow版本问题报错)
    tf.compat.v1.disable_eager_execution()
    # 指定已下载的ELMo模型文件夹路径(建议使用模型的相对路径)
    elmo_model_path = 'D:/桌面/Bert_Q&A/model'
    # 加载本地下载好的模型
    elmo = hub.Module(elmo_model_path, trainable=True)
    # 三条英文测试数据
    sentence_lists = ["My name is Lu ZhangWen",
                      "I am studying natural language processing",
                      "I really love it"]
    # 通过预训练模型处理测试数据
    output = elmo(sentence_lists, as_dict=True)
    print(output)
    # 启动图运行elmo模型得到三句测试数据词向量
    # 这里调用tf.compat.v1是为了调用tensorflow1.x中的函数
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    elmo_embedding = sess.run(output['elmo'])
    print(elmo_embedding)
