# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# import torch


def split(data, num_split=5):
    train = []
    val   = []
    for i in range(num_split):
        val_indices = range(i, len(data), num_split)
        train_indices = [idx for idx in range(len(data)) if idx not in val_indices]

        val.append(data.iloc[val_indices])
        train.append(data.iloc[train_indices])

    return train, val

def norml(data, rate=0.005, pred=False):
    for col in data.columns:
        try:
            y = list(data[col])
            y = sorted(y)
            a = round(len(y)*rate)
            min_data = y[a]
            max_data = y[-a]
            if max_data != min_data:
                data.loc[data[col]>max_data, col] = max_data
                data.loc[data[col]<min_data, col] = min_data
                if not col in ['Ehull', 'd-band center', 'p-band center', 'EV', 'EH', 'OverlappingArea', 'OverlappingCenter', 'Polarization Resistance', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Volume', 'ShrinkageV','FreeVolume','SymmetryOperations']:
                    data[col] = (data[col] - min_data) / (max_data - min_data)
                #if not pred:
                #    data[col] = (data[col]-min_data)/(max_data-min_data)
                #else:
                #    if not col in ['p-band center', 'EV', 'EH', 'OverlappingArea','OverlappingCenter', 'Polarization Resistance', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Volume', 'ShrinkageV','FreeVolume','SymmetryOperations']:
                #        data[col] = (data[col] - min_data) / (max_data - min_data)
            elif max_data != 0:
                data[col] = data[col]/max_data
            else:
                pass
        except:
            # print(col, ' not normlized')
            continue
    return data

def eval_score(gt, pred):
    # 评估函数，计算均方误差
    mse = mean_squared_error(gt, pred)
    return mse

def get_col(data, args):
    col = data.columns
    if args.stage == 1:
        feat_col = list(col)[5:155]
    elif args.stage == 2:
        feat_col = list(col)[5:175]+list(col)[177:194]
    elif args.stage == 2.1:
        feat_col = list(col)[5:155]
    else:
        print('Stage error')
        quit()

    if args.targ in ['1', '2', '3', '4']:
        args.targ = int(args.targ)

    if isinstance(args.targ, int):
        if args.stage == 1:
            target = col[args.targ + 154]
        else:
            target = col[args.targ + 193]
    elif isinstance(args.targ, str):
        target = args.targ
    else:
        print('Target type error')
    print('Target: ', target)
    targ_col = [target]

    try:
        targ = data[targ_col]
    except:
        print('Target col don\'t exist, check your input')
        quit()
    return feat_col, targ_col

def get_data(args):
    data = pd.read_csv(args.train)
    data = norml(data, pred=args.pred)

    feat_col, targ_col = get_col(data, args)

    train, val = split(data, num_split=args.num_split)
    data = {'train': train, 'val': val}

    return data, feat_col, targ_col