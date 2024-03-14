# Edited by ZhangWen Lu
# Date:2024.1.1
# Env: tensorflow 2.12.0
#      pytorch 1.12.1

import pandas as pd

# 加载斯坦福大学提供的CoQA数据集
file = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')
file.head()
del file["version"]

# 只保留text(上下文)、question(问题)、answer(答案)三项
cols = ["text", "question", "answer"]
comp_list = []
for index, row in file.iterrows():
    for i in range(len(row["data"]["questions"])):
        temp_list = [row["data"]["story"], row["data"]["questions"][i]["input_text"],
                     row["data"]["answers"][i]["input_text"]]
        comp_list.append(temp_list)
new_df = pd.DataFrame(comp_list, columns=cols)

# 将获取的文件保存为csv文件
new_df.to_csv("CoQA_data.csv", index=False)
