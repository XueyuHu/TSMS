import pandas as pd
import numpy as np
from tqdm import tqdm

from dataset import *
import warnings
# warnings.filterwarnings("ignore")

import joblib

from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression

import argparse

def train_with_cross_validation(data, feat_col, targ_col, model):
    scores = []

    for i, train in enumerate(tqdm(data['train'])):
        val = data['val'][i]
        train_x = train[feat_col].astype('float64')
        train_y = train[targ_col].astype('float64')

        val_x = val[feat_col].astype('float64')
        val_y = val[targ_col].astype('float64')

        model.fit(train_x, train_y)
        pred = model.predict(val_x)
        s = eval_score(val_y, pred)
        scores.append(s)

    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-stage', default=1, type=int, help='stage')
    parser.add_argument('-num_split', default=5, type=int, help='num split')
    parser.add_argument('-seed', default=2023, type=int, help='random seed')
    parser.add_argument('-train', default='Data/dataset_1.csv', type=str, help='input path')
    parser.add_argument('-targ', default=4, help='target either int or str')
    parser.add_argument('-test', default=False, help='Data/test_1.csv')
    parser.add_argument('-pth', default=False, help='Checkpoint output path')

    parser.add_argument('-alpha', default=0.1, help='Lasso alpha')

    args = parser.parse_args()

    np.random.seed(args.seed)

    model = Lasso(alpha=args.alpha)
    # model = LinearRegression()

    if args.test:
        test = pd.read_csv(args.test)
        test = norml(data=test, test=True)
        feat_col, targ_col = get_col(test, args)


        train = pd.read_csv(args.train)
        train = norml(data=train, test=True)
        train_x = train[feat_col].astype('float64')
        train_y = train[targ_col].astype('float64')
        model.fit(train_x, train_y)

        test_x = test[feat_col].astype('float64')
        test_y = test[targ_col].astype('float64')
        pred_y = model.predict(test_x)
        mse = eval_score(test_y, pred_y)
        print('MSE on Test Set:', mse)

        quit()

    if args.train:
        data, feat_col, targ_col = get_data(args)
        scores = train_with_cross_validation(data, feat_col, targ_col, model=model)

        avg_score = np.average(scores)
        print('Cross-validation Scores:', scores)
        print('Average Score:', avg_score)

