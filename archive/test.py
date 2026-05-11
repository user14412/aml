import pandas as pd

data_dir = "../TableShift Dataset/acsfoodstamps/"

X_train = pd.read_csv(data_dir + "acsfoodstamps_Xtrain.csv")
y_train = pd.read_csv(data_dir + "acsfoodstamps_ytrain.csv")

print("训练集特征的形状:", X_train.shape) # 看看有多少行样本，多少列特征
print("训练集标签的形状:", y_train.shape)
print("\n前5行特征数据：")
print(X_train.head())