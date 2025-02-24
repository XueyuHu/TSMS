import sys

import numpy as np
import pandas as pd
from numpy.random import randint
import random
import os
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Add, Dense, Flatten, Input, Lambda, BatchNormalization, Dropout, concatenate
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error
from keras.backend import clear_session
from keras.models import clone_model
from tqdm import tqdm

from dataset import *
from model import *
import argparse

tf.keras.backend.set_floatx('float64')


def train_with_cross_validation(data, feat_col, targ_col, model, args):
    scores = []

    m = model
    m.build((None, len(feat_col)))
    m.compile(optimizer=Adam(learning_rate=args.lr), loss='mse', metrics=['mse'])
    m.save_weights('Result/ann_initial.h5')

    for i, train in enumerate(tqdm(data['train'])):

        val = data['val'][i]
        train_x = train[feat_col].astype('float64')
        train_y = train[targ_col].astype('float64')

        val_x = val[feat_col].astype('float64')
        val_y = val[targ_col].astype('float64')

        train_y = train_y.T.iloc[0]
        val_y = val_y.T.iloc[0]

        m.load_weights('Result/ann_initial.h5')

        m.fit(
            train_x,
            train_y,
            validation_data=(val_x, val_y),
            epochs=args.e,
            verbose=args.v)

        s = m.evaluate(
            val_x,
            val_y,
            verbose=0
        )
        scores.append(s[1])

    return scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-stage', default=2.2, type=int, help='stage')
    parser.add_argument('-num_split', default=5, type=int, help='num split')
    parser.add_argument('-seed', default=2022, type=int, help='random seed')
    parser.add_argument('-train', default='Data/train_2.csv', type=str, help='input path')
    parser.add_argument('-targ', default=1, help='target either int or str')
    parser.add_argument('-test', default='Data/test_2.csv', help='Data/test_1.csv')
    parser.add_argument('-pred', default=True, help='Normalize or not')
    parser.add_argument('-drop', default=False, help='Drop or not')
    parser.add_argument('-pth', default=False, help='Checkpoint output path')
    parser.add_argument('-h5', default=False)
    parser.add_argument('-v', default=1, help='verbose')

    parser.add_argument('-e', default=300, type=int, help='Epoch')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')


    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


    if args.test:
        test = pd.read_csv(args.test)
        feat_col, targ_col = get_col(test, args)

        model = ANN_s1()
        model.build((None, len(feat_col)))

        try:
            checkpoint_path = 'Result/' + args.h5
            model.load_weights(checkpoint_path)
            print(args.h5, ' loaded')
        except:
            train = pd.read_csv(args.train)

            if args.drop:
                on = ['Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5', 'Ratio_1', 'Ratio_2', 'Ratio_3', 'Ratio_4', 'Ratio_5']

                common_rows = pd.merge(train, test, how='inner', on=on)
                train = train[~train[on].apply(tuple, axis=1).isin(common_rows[on].apply(tuple, axis=1))]

            merged_df = pd.concat([train, test], ignore_index=True)
            merged_df = norml(data=merged_df, pred=args.pred)
            train = merged_df[:len(train)]
            test = merged_df[len(train):]

            train_x = train[feat_col].astype('float64')
            train_y = train[targ_col].astype('float64').T.iloc[0]

            model.compile(optimizer=Adam(learning_rate=args.lr), loss='mse', metrics=['mse'])

            model.fit(
                train_x,
                train_y,
                epochs=args.e,
                verbose=args.v)


        test_x = test[feat_col].astype('float64')
        test_y = test[targ_col].astype('float64')
        pred = model.predict(test_x)
        mse = eval_score(test_y, pred)
        print('MSE on Test Set:', mse)
        sys.exit()

    if args.train:
        data, feat_col, targ_col = get_data(args)
        model = ANN_s1()

        scores = train_with_cross_validation(data, feat_col, targ_col, model, args)

        avg_score = np.average(scores)
        print('Cross-validation Scores:', scores)
        print('Average Score:', avg_score)

    if args.pth:
        checkpoint_path = 'Result/' + args.pth
        model.set_weights(weight)
        model.save_weights(checkpoint_path, save_format='h5')