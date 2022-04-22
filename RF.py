from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.rcParams['font.family']='STSong'

import pandas as pd
import numpy as np

CSV_FILE_PATH = "C:/Users/10713/Desktop/钢材性能预测/"
file_name = "a_use.CSV"
df = pd.read_csv(CSV_FILE_PATH + file_name, encoding = "gbk")
data = df[["实验屈服值","实验抗拉值","实验伸长率","出炉温度","加热时间","板坯厚度","中间坯厚度","粗轧压下率",
           "精轧开轧温度","终轧温度","成品厚度","精轧压缩比","粗轧压缩比",
           "C","Si","Mn","AlT","Nb","V","Ti","Ni","Cu","Cr","Mo","P","S","Ceq","Pcm","温度差"]]
print(data)

file_name_test = "test.csv"
df_test = pd.read_csv(CSV_FILE_PATH + file_name_test, encoding = "gbk")
data_test = df_test[["实验屈服值","实验抗拉值","实验伸长率","出炉温度","加热时间","板坯厚度","中间坯厚度","粗轧压下率",
                     "精轧开轧温度","终轧温度","成品厚度","精轧压缩比","粗轧压缩比",
                     "C","Si","Mn","AlT","Nb","V","Ti","Ni","Cu","Cr","Mo","P","S","Ceq","Pcm","温度差"]]
print(data_test)

X_col = ["出炉温度","加热时间","板坯厚度","中间坯厚度","粗轧压下率",
         "精轧开轧温度","终轧温度","成品厚度","精轧压缩比","粗轧压缩比",
         "C","Si","Mn","AlT","Nb","V","Ti","Ni","Cu","Cr"]
y_col = ["实验屈服值"]

X_train = data[X_col].values
y_train = data[y_col].values

X_test = data_test[X_col].values
y_test = data_test[y_col].values
scores = []

n = 16
r = 99
for k in range (1, r):
    reg = RandomForestRegressor(max_depth=25,
                               min_samples_split=2,
                               criterion="squared_error",
                               min_samples_leaf=1,
                               min_weight_fraction_leaf=0.0,
                               max_features="auto",
                               max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               bootstrap=True,
                               oob_score=False,
                               n_jobs=None,
                               random_state=4,
                               warm_start=True,
                               ccp_alpha=0.0,
                               max_samples=k/100)
    start = time.time()
    reg.fit(X_train, y_train.ravel())
    finish = time.time()

    predict = reg.predict(X_test)

    score = reg.score(X_test, y_test.ravel())
    scores.append(score)

predict = reg.predict(X_test)
plt.plot(X_test[:,0], predict, 'r^', label = 'original data')
plt.xlabel("中间坯厚度")
plt.ylabel("实验屈服值")
plt.title("预测值")
plt.show()
plt.plot(X_test[:,0], y_test, 'bo', label = 'original data')
plt.xlabel("中间坯厚度")
plt.ylabel("实验屈服值")
plt.title("实际值")
plt.show()
print(score, finish-start)

plt.plot(np.linspace(1,r-1,r-1), scores, 'r^', label = 'original data')
plt.ylabel("sample%")
plt.xlabel("R^2")
plt.title("")
plt.show()
