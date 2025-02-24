import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataset import *
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
# import lightgbm as lgb
# import catboost as cab
from joblib import dump, load

from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

import shap

import argparse

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def train_with_cross_validation(data, feat_col, targ_col, model, args):
    dump(model, 'Result/'+args.model+'_initial.joblib')

    scores = []

    for i, train in enumerate(tqdm(data['train'])):
        model = load('Result/'+args.model+'_initial.joblib')
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

    parser.add_argument('-model', default='xgb', type=str, help='model')
    parser.add_argument('-stage', default=1, type=int, help='stage')
    parser.add_argument('-num_split', default=5, type=int, help='num split')
    parser.add_argument('-seed', default=2023, type=int, help='random seed')
    parser.add_argument('-train', default='Data/train_1.csv', type=str, help='input path')
    parser.add_argument('-targ', default=1, help='target either int or str')
    parser.add_argument('-test', default='Data/test_1.csv', help='Data/test_1.csv')
    parser.add_argument('-pred', default=False, help='Normalize or not')
    parser.add_argument('-drop', default=False, help='Drop or not')
    parser.add_argument('-shap', default=False, help='Shap value analysis')
    parser.add_argument('-pth', default=False, help='Checkpoint output path')

    parser.add_argument('-depth', default=False, help='max_depth')
    parser.add_argument('-leaves', default=False, help='max_leaves')
    parser.add_argument('-child', default=False, help='min child weight')
    parser.add_argument('-lr', default=False, help='learning rate')
    parser.add_argument('-n', default=False, help='n estimators')
    parser.add_argument('-parm', default=False)

    args = parser.parse_args()

    np.random.seed(args.seed)

    params = {
        'max_depth'       : 4,
        'learning_rate'   : 0.01,
        'n_estimators'    : 200,
        'max_leaves'      : 0,
        'min_child_weight': 20
    }
    params_2 = {
        'kernel'          : 'poly',
        'degree'          : 10,
        'gamma'           : 'auto',
        'tol'             : 0.001,
        'epsilon'         : 0.1,
        'shrinking'       : True,
        'cache_size'      : 200,
        'verbose'         : False,
        'max_iter'        : -1
    }
    if args.depth:
        params['max_depth'] = args.depth
    if args.child:
        params['min_child_weight'] = args.child
    if args.lr:
        params['learning_rate'] = args.lr
    if args.leaves:
        params['max_leaves'] = args.leaves
    if args.n:
        params['n_estimators'] = args.n

    loaded_model = False
    if args.model == 'xgb':
        model = xgb.XGBRegressor(**params)
    elif args.model == 'rf':
        model = xgb.XGBRFRegressor(**params)
    elif args.model == 'lgb':
        model = lgb.LGBMRegressor(**params)
    elif args.model == 'svr':
        model = SVR(**params_2)
    elif args.model == 'linear':
        model = linear_model.LinearRegression()
    else:
        print('Trying to load model from', args.model)
        model = load(args.model)
        print('Model loaded')
        loaded_model = True


    if args.parm:

        best_score = 99999

        params_list = {
            'max_depth': [4, 8, 16],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [200, 400],
            'max_leaves': [0, 10, 20],
            'min_child_weight': [0, 10, 20],
            'seed': [2022]
        }

        data, feat_col, targ_col = get_data(args)

        for se_ in params_list['seed']:
            np.random.seed(se_)
            for d_ in params_list['max_depth']:
                for lr_ in params_list['learning_rate']:
                    for n_ in params_list['n_estimators']:
                        for le_ in params_list['max_leaves']:
                            for ch_ in params_list['min_child_weight']:
                                params = {
                                    'max_depth': d_,
                                    'learning_rate': lr_,
                                    'n_estimators': n_,
                                    'max_leaves': le_,
                                    'min_child_weight': ch_
                                }

                                model = xgb.XGBRegressor(**params)
                                scores = train_with_cross_validation(data, feat_col, targ_col, model=model, args=args)

                                avg_score = np.average(scores)
                                print('Average Score:', avg_score)

                                if best_score > avg_score:
                                    best_seed = se_
                                    best_score = avg_score
                                    best_params = params

        print('Best score:', best_score)
        print(best_params)
        print('Best seed:', best_seed)
        sys.exit()


    if args.test:
        test = pd.read_csv(args.test)
        feat_col, targ_col = get_col(test, args)

        if not loaded_model:
            try:
                checkpoint_path = 'Result/' + args.pth + '.pth'
            except:
                checkpoint_path = 'Result/_model.pth'
            train = pd.read_csv(args.train)

            if args.drop:
                on = ['Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5']

                common_rows = pd.merge(train, test, how='inner', on=on)
                train = train[~train[on].apply(tuple, axis=1).isin(common_rows[on].apply(tuple, axis=1))]

            merged_df = pd.concat([train, test], ignore_index=True)
            merged_df = norml(data=merged_df, pred=args.pred)
            train = merged_df[:len(train)]
            test = merged_df[len(train):]
            print('Train size:', len(train), '  Test size:', len(test))

            train_x = train[feat_col].astype('float64')
            train_y = train[targ_col].astype('float64')

            model.fit(train_x, train_y)

        test_x = test[feat_col].astype('float64')
        test_y = test[targ_col].astype('float64')
        pred_y = model.predict(test_x)
        mse = eval_score(test_y, pred_y)
        print('MSE on Test Set:', mse)
        try:
            pred_path = 'Result/' + args.pth + '_pred.csv'
            imp_path = 'Result/' + args.pth + '_imp.csv'
        except:
            pred_path = 'Result/_pred.csv'
            imp_path = 'Result/_imp.csv'
        targ = targ_col[0]
        pred_df = test[['Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5', 'Ratio_1', 'Ratio_2', 'Ratio_3', 'Ratio_4', 'Ratio_5', targ]]
        pred_df['Prediction'] = pred_y
        pred_df.to_csv(pred_path, index=False)
        print('Prediction saved to', pred_path)

        # model_imp = model.feature_importances_
        # perm_imp = permutation_importance(model, test_x, test_y, n_repeats=10, random_state=0)
        # shap_imp = shap.TreeExplainer(model).shap_values(test_x)
        # imp = pd.DataFrame([feat_col, model_imp, perm_imp.importances_mean, shap_imp.mean(0)])
        # imp.T.to_csv(imp_path, header=None, index=0)

        if args.shap:
            shap.initjs()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(test_x)

            # Calculate the mean absolute SHAP values for each feature
            # Extract actual SHAP values from the Explanation object
            actual_shap_values = shap_values.values

            # Calculate the mean absolute SHAP values for each feature
            mean_abs_shap_values = np.mean(np.abs(actual_shap_values), axis=0)

            # Proceed as before to sort the features and print/save the values
            # (The rest of the code remains the same)

            # Sort the features based on the mean absolute SHAP values
            sorted_feature_indices = np.argsort(-mean_abs_shap_values)  # Negative for descending order
            sorted_features = [feat_col[i] for i in
                               sorted_feature_indices]  # Assuming feat_col is the list of feature names
            sorted_mean_abs_shap_values = mean_abs_shap_values[sorted_feature_indices]

            # Print the feature names alongside their mean absolute SHAP values
            print("Feature | Mean Absolute SHAP Value")
            for feature, value in zip(sorted_features, sorted_mean_abs_shap_values):
                print(f"{feature}: {value}")

            # Optionally, save this information to a text file
            with open("shap_values_summary.txt", "w") as file:
                file.write("Feature | Mean Absolute SHAP Value\n")
                for feature, value in zip(sorted_features, sorted_mean_abs_shap_values):
                    file.write(f"{feature}: {value}\n")
            # for v_ in shap_values:
            #     shap.plots.waterfall(v_)
            shap.plots.force(shap_values[0])
            # shap.plots.scatter(shap_values[:,"ionizationEnergy_6"], color=shap_values[:,"ionizationEnergy_5"])
            plt.figure(dpi=300)
            shap.plots.beeswarm(shap_values, max_display=6)
            #plt.savefig('high_resolution_beeswarm.png',dpi=300)
            plt.figure(dpi=300)
            shap.plots.bar(shap_values, max_display=6)

        sys.exit()


    if args.train:
        data, feat_col, targ_col = get_data(args)
        np.random.seed(args.seed)
        scores = train_with_cross_validation(data, feat_col, targ_col, model=model, args=args)

        avg_score = np.average(scores)
        print('Cross-validation Scores:', scores)
        print('Average Score:', avg_score)




