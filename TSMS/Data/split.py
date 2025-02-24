import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def split(data, num=5):
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for i in range(data.shape[0]):
        if i % num == 1:
            # 将行添加到测试集
            test_set = pd.concat([test_set, data.loc[i]], axis=1)

        else:
            # 将行添加到训练集
            train_set = pd.concat([train_set, data.loc[i]], axis=1)
    return train_set.T, test_set.T

def split_KFold(data, num=5):
    kf = KFold(n_splits=num, shuffle=False)
    datasets = []

    for train_index, _ in kf.split(data):
        subset = data.loc[train_index]
        datasets.append(subset)

    return datasets

if __name__ == '__main__':
    data = pd.read_csv('dataset_1_La.csv')

    # subsets = split_KFold(data)
    #
    # for i, subset in enumerate(subsets):
    #     subset.to_csv('dataset_1_{}.csv'.format(i), index=0)


    train, test = split(data)
    train.to_csv('train_1_La.csv', index=0)
    test.to_csv('test_1_La.csv', index=0)

