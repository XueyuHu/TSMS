import sys
import json
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm

from dataset_v2 import *
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
# import lightgbm as lgb
# import catboost as cab
from joblib import dump, load

from sklearn import preprocessing, linear_model
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

import shap

import argparse

from matplotlib import pyplot as plt
import matplotlib
# Safe backend for servers/CI (no GUI required)
matplotlib.use('Agg')


def _maybe_fold_safe_scale(train_x, val_x, feat_col, args):
    """Fit on train only; transform both train and val/test to avoid leakage."""
    if not getattr(args, 'pred', False):
        return train_x, val_x, None

    skip_cols = {
        'Ehull', 'd-band center', 'p-band center', 'EV', 'EH',
        'OverlappingArea', 'OverlappingCenter', 'Polarization Resistance',
        'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Volume', 'ShrinkageV',
        'FreeVolume', 'SymmetryOperations'
    }
    scaler = ClipMinMaxScaler(rate=getattr(args, 'clip_rate', 0.005), skip_cols=skip_cols)
    train_x_scaled = scaler.fit_transform(train_x, feat_col)
    val_x_scaled = scaler.transform(val_x)
    return train_x_scaled, val_x_scaled, scaler


def train_with_cross_validation(data, feat_col, targ_col, model, args):
    """Cross-validation training loop (fold-safe preprocessing + deterministic)."""
    os.makedirs('Result', exist_ok=True)

    # Save feature/target metadata for reproducibility (helps reviewers)
    meta = {
        'model': args.model,
        'stage': args.stage,
        'num_split': args.num_split,
        'seed': args.seed,
        'n_jobs': args.n_jobs,
        'pred_fold_safe_scale': bool(args.pred),
        'group_cv': bool(getattr(args, 'group_cv', False)),
        'clip_rate': args.clip_rate,
        'feat_col': list(feat_col),
        'targ_col': str(targ_col),
    }
    with open('Result/run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    dump(model, 'Result/' + args.model + '_initial.joblib')

    scores = []

    for i, train in enumerate(tqdm(data['train'])):
        model = load('Result/' + args.model + '_initial.joblib')
        val = data['val'][i]

        train_x = train[feat_col].astype('float64')
        train_y = train[targ_col].astype('float64').values.ravel()

        val_x = val[feat_col].astype('float64')
        val_y = val[targ_col].astype('float64').values.ravel()

        train_x, val_x, _ = _maybe_fold_safe_scale(train_x, val_x, feat_col, args)

        model.fit(train_x, train_y)
        pred = model.predict(val_x)
        s = eval_score(val_y, pred)
        scores.append(s)

    return scores


def _save_shap_artifacts(model, X: pd.DataFrame, feat_col, out_prefix: str = "Result/shap"):
    """Generate SHAP summary + plots, saved to disk (non-interactive)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)  # shap.Explanation

    # Mean absolute SHAP values
    actual = shap_values.values
    mean_abs = np.mean(np.abs(actual), axis=0)
    idx = np.argsort(-mean_abs)
    sorted_features = [feat_col[i] for i in idx]
    sorted_values = mean_abs[idx]

    txt_path = f"{out_prefix}_values_summary.txt"
    with open(txt_path, "w") as f:
        f.write("Feature | Mean Absolute SHAP Value\n")
        for feature, value in zip(sorted_features, sorted_values):
            f.write(f"{feature}: {value}\n")

    # Beeswarm
    plt.figure(dpi=300)
    shap.plots.beeswarm(shap_values, max_display=min(20, len(feat_col)), show=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_beeswarm.png", dpi=300)
    plt.close()

    # Bar
    plt.figure(dpi=300)
    shap.plots.bar(shap_values, max_display=min(20, len(feat_col)), show=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bar.png", dpi=300)
    plt.close()

    # Force plot (first sample) -> save html (optional)
    try:
        force = shap.plots.force(shap_values[0], matplotlib=False)
        shap.save_html(f"{out_prefix}_force_0.html", force)
    except Exception:
        pass

    print("SHAP artifacts saved:")
    print(" -", txt_path)
    print(" -", f"{out_prefix}_beeswarm.png")
    print(" -", f"{out_prefix}_bar.png")
    print(" -", f"{out_prefix}_force_0.html (if generated)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-model', default='xgb', type=str, help='model: xgb/rf/svr/linear or a joblib path')
    parser.add_argument('-stage', default=1, type=int, help='stage (used for feature selection in get_col)')
    parser.add_argument('-num_split', default=5, type=int, help='number of CV folds')
    parser.add_argument('-seed', default=2023, type=int, help='random seed')
    parser.add_argument('-n_jobs', default=1, type=int, help='threads for xgboost (set 1 for determinism)')
    parser.add_argument('-clip_rate', default=0.005, type=float, help='clip rate for fold-safe scaling when -pred is enabled')
    parser.add_argument('-train', default='Data/train_1.csv', type=str, help='train csv path')
    parser.add_argument('-targ', default=1, type=str, help='target column (index or name)')
    parser.add_argument('-test', default='Data/test_1.csv', type=str, help='test csv path')

    # boolean flags (reviewer-proof; avoids string truthiness bugs)
    parser.add_argument('-pred', action='store_true', help='Enable fold-safe normalization (fit on train fold only)')
    parser.add_argument('-drop', action='store_true', help='Drop exact overlap between train/test by composition columns')
    parser.add_argument('-group_cv', action='store_true', help='Use GroupKFold by composition key to reduce composition-level leakage')
    parser.add_argument('-shap', action='store_true', help='Run SHAP analysis on test set (TreeExplainer)')
    parser.add_argument('-parm', action='store_true', help='Run hyperparameter grid search')
    parser.add_argument('-pth', default='', type=str, help='Output prefix for saving artifacts (optional)')

    # optional hyperparameter overrides
    parser.add_argument('-depth', default=None, type=int, help='max_depth')
    parser.add_argument('-leaves', default=None, type=int, help='max_leaves')
    parser.add_argument('-child', default=None, type=float, help='min_child_weight')
    parser.add_argument('-lr', default=None, type=float, help='learning rate')
    parser.add_argument('-n', default=None, type=int, help='n_estimators')

    args = parser.parse_args()

    # Parse target: allow passing an integer index as a string
    if isinstance(args.targ, str) and args.targ.isdigit():
        args.targ = int(args.targ)


    np.random.seed(args.seed)
    random.seed(args.seed)

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
    if args.depth is not None:
        params['max_depth'] = int(args.depth)
    if args.child is not None:
        params['min_child_weight'] = float(args.child)
    if args.lr is not None:
        params['learning_rate'] = float(args.lr)
    if args.leaves is not None:
        params['max_leaves'] = int(args.leaves)
    if args.n is not None:
        params['n_estimators'] = int(args.n)

    loaded_model = False
    if args.model == 'xgb':
        params.setdefault('random_state', args.seed)
        params.setdefault('n_jobs', args.n_jobs)
        model = xgb.XGBRegressor(**params)
    elif args.model == 'rf':
        params.setdefault('random_state', args.seed)
        params.setdefault('n_jobs', args.n_jobs)
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

    # -------------------- Hyperparameter search --------------------
    if args.parm:

        best_score = 99999

        params_list = {
            'max_depth': [4, 8, 16],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [200, 500],
            'max_leaves': [0, 16, 64],
            'min_child_weight': [0, 10, 20],
            'seed': [2022]
        }

        data, feat_col, targ_col = get_data(args)

        for se_ in params_list['seed']:
            np.random.seed(se_)
            for d_ in params_list['max_depth']:
                for lr_ in params_list['learning_rate']:
                    for n_ in params_list['n_estimators']:
                        for l_ in params_list['max_leaves']:
                            for c_ in params_list['min_child_weight']:

                                params = {
                                    'max_depth'       : d_,
                                    'learning_rate'   : lr_,
                                    'n_estimators'    : n_,
                                    'max_leaves'      : l_,
                                    'min_child_weight': c_,
                                    'random_state'    : args.seed,
                                    'n_jobs'          : args.n_jobs
                                }

                                if args.model == 'xgb':
                                    model = xgb.XGBRegressor(**params)
                                elif args.model == 'rf':
                                    model = xgb.XGBRFRegressor(**params)

                                scores = train_with_cross_validation(data, feat_col, targ_col, model, args)
                                score = sum(scores) / len(scores)

                                if score < best_score:
                                    best_score = score
                                    best_params = params
                                    print('Best score:', best_score)
                                    print('Best params:', best_params)

        print('Final best score:', best_score)
        print('Final best params:', best_params)
        sys.exit()

    # -------------------- Test (prediction + optional SHAP) --------------------
    if args.test:
        test = pd.read_csv(args.test)
        feat_col, targ_col = get_col(test, args)

        if not loaded_model:
            train = pd.read_csv(args.train)

            # Optional: drop exact overlap between train/test
            if args.drop:
                # stricter than original: include ratios if available
                on = ['Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5',
                      'Ratio_1', 'Ratio_2', 'Ratio_3', 'Ratio_4', 'Ratio_5']
                if all(c in train.columns for c in on) and all(c in test.columns for c in on):
                    common_rows = pd.merge(train, test, how='inner', on=on)
                    train = train[~train[on].apply(tuple, axis=1).isin(common_rows[on].apply(tuple, axis=1))]
                else:
                    on2 = ['Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5']
                    common_rows = pd.merge(train, test, how='inner', on=on2)
                    train = train[~train[on2].apply(tuple, axis=1).isin(common_rows[on2].apply(tuple, axis=1))]

            train_x = train[feat_col].astype('float64')
            train_y = train[targ_col].astype('float64').values.ravel()

            test_x = test[feat_col].astype('float64')
            test_y = test[targ_col].astype('float64').values.ravel()

            # fold-safe scaling: fit on train, apply to test
            train_x, test_x, _ = _maybe_fold_safe_scale(train_x, test_x, feat_col, args)

            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)

        else:
            test_x = test[feat_col].astype('float64')
            test_y = test[targ_col].astype('float64').values.ravel()
            pred_y = model.predict(test_x)

        mse = eval_score(test_y, pred_y)
        print('MSE on Test Set:', mse)

        # Output paths
        prefix = f"Result/{args.pth}" if args.pth else "Result/_model"
        pred_path = prefix + "_pred.csv"

        targ = targ_col[0]
        keep_cols = [c for c in ['Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5',
                                 'Ratio_1', 'Ratio_2', 'Ratio_3', 'Ratio_4', 'Ratio_5', targ]
                     if c in test.columns]
        pred_df = test[keep_cols].copy()
        pred_df['Prediction'] = pred_y
        pred_df.to_csv(pred_path, index=False)
        print('Prediction saved to', pred_path)

        if args.shap:
            out_prefix = prefix + "_shap"
            _save_shap_artifacts(model, pd.DataFrame(test_x, columns=feat_col), feat_col, out_prefix=out_prefix)

        sys.exit()

    # -------------------- Train (CV) --------------------
    if args.train:
        data, feat_col, targ_col = get_data(args)
        scores = train_with_cross_validation(data, feat_col, targ_col, model=model, args=args)

        avg_score = np.average(scores)
        print('Cross-validation Scores:', scores)
        print('Average Score:', avg_score)
