# Loading Imports
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.rcParams['font.family']='STSong'

import pandas as pd
import numpy as np

# Loading Datasets

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

# Preparing Train and Test datasets

# The Features you want to learn
X_col = ["出炉温度","加热时间","板坯厚度","中间坯厚度","粗轧压下率",
         "精轧开轧温度","终轧温度","成品厚度","精轧压缩比","粗轧压缩比",
         "C","Si","Mn","AlT","Nb","V","Ti","Ni","Cu","Cr"]

# The Result you want to reach
y_col = ["实验屈服值"]

X_train = data[X_col].values
y_train = data[y_col].values

X_test = data_test[X_col].values
y_test = data_test[y_col].values

# Define Model Properties
n = 16
reg = GradientBoostingRegressor(random_state = 19082,
                                n_estimators = n,
                                loss = "squared_error",
                                subsample = 0.40,
                                criterion = "friedman_mse",
                                min_samples_split = 2,
                                min_samples_leaf = 1,
                                min_weight_fraction_leaf = 0.0,
                                max_depth = 10,
                                min_impurity_decrease = 0.0,
                                init = None,
                                max_features = None,
                                warm_start = False,
                                ccp_alpha = 0.0)
start = time.time()

# Train Model
reg.fit(X_train, y_train.ravel())

finish = time.time()

# Feature importance graph
feature_importance = reg.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_col)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

# Learning score graph
plt.plot(np.linspace(1,n,n), reg.train_score_, 'b-', label = 'original data')
plt.xlabel("n")
plt.ylabel("score")
plt.show()

# Prediciton and score
predict = reg.predict(X_test)

# Score and Efficiency in s
score = reg.score(X_test, y_test.ravel())

# Prediction Result Visualized
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
