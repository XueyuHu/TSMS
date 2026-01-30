# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold
# import torch



def split(data, num_split=5, shuffle=True, seed=2023, groups=None):
    """K-fold split with optional grouping (GroupKFold).

    Parameters
    ----------
    data : pd.DataFrame
    num_split : int
        Number of folds.
    shuffle : bool
        Whether to shuffle rows before splitting. For GroupKFold, shuffling is applied
        to the dataframe (and groups) *before* splitting to reduce order effects.
    seed : int
        Random seed used for shuffling (and KFold shuffling).
    groups : array-like or None
        If provided, GroupKFold is used so that the same group never appears in both
        train and validation within a fold.

    Returns
    -------
    train_folds, val_folds : list[pd.DataFrame], list[pd.DataFrame]
    """
    if num_split < 2:
        raise ValueError("num_split must be >= 2")

    df = data.copy()
    if groups is not None:
        groups = pd.Series(groups).reset_index(drop=True)

    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if groups is not None:
            groups = groups.loc[df.index].reset_index(drop=True)

    train_folds, val_folds = [], []
    if groups is not None:
        splitter = GroupKFold(n_splits=num_split)
        split_iter = splitter.split(df, groups=groups)
    else:
        splitter = KFold(n_splits=num_split, shuffle=shuffle, random_state=seed if shuffle else None)
        split_iter = splitter.split(df)

    for train_idx, val_idx in split_iter:
        train_folds.append(df.iloc[train_idx].reset_index(drop=True))
        val_folds.append(df.iloc[val_idx].reset_index(drop=True))

    return train_folds, val_folds


def norml(data, rate=0.005, pred=False, inplace=True):
    """Clip extreme values by quantiles and (optionally) min-max normalize.

    DEPRECATED (for CV workflows)
    ---------------------------
    Prefer ClipMinMaxScaler + fold-safe scaling inside the CV loop.

    WARNING
    -------
    This function can cause data leakage if you apply it to the full dataset
    before cross-validation. Prefer the ClipMinMaxScaler below to fit on the
    training fold only.

    Parameters
    ----------
    rate : float
        Fraction to clip on each tail (e.g., 0.005 clips 0.5% low and high).
    inplace : bool
        If False, operates on a copy and returns it.
    """
    df = data if inplace else data.copy()

    for col in df.columns:
        try:
            y = sorted(df[col].astype(float).tolist())
            n = len(y)
            if n == 0:
                continue

            a = int(round(n * rate))
            # Guard: when a == 0, use full range [0, -1]
            lo_idx = min(max(a, 0), n - 1)
            hi_idx = n - 1 if a == 0 else max(min(n - a - 1, n - 1), 0)

            min_data = y[lo_idx]
            max_data = y[hi_idx]

            if max_data != min_data:
                df.loc[df[col] > max_data, col] = max_data
                df.loc[df[col] < min_data, col] = min_data

                # Keep certain physical/label-like columns unnormalized
                skip_cols = {
                    'Ehull', 'd-band center', 'p-band center', 'EV', 'EH',
                    'OverlappingArea', 'OverlappingCenter', 'Polarization Resistance',
                    'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Volume', 'ShrinkageV',
                    'FreeVolume', 'SymmetryOperations'
                }
                if col not in skip_cols:
                    df[col] = (df[col] - min_data) / (max_data - min_data)

            elif max_data != 0:
                df[col] = df[col] / max_data
            else:
                pass
        except Exception:
            # silently skip non-numeric columns
            continue

    return df


class ClipMinMaxScaler:
    """Fold-safe clip + min-max scaling for pandas DataFrames.

    Fit on train fold, transform both train/val/test to avoid leakage.
    """

    def __init__(self, rate=0.005, skip_cols=None):
        self.rate = float(rate)
        self.skip_cols = set(skip_cols or [])
        self._fitted_cols = []
        self._min = {}
        self._max = {}

    def fit(self, df: pd.DataFrame, cols):
        self._fitted_cols = list(cols)
        for col in self._fitted_cols:
            if col in self.skip_cols:
                continue
            s = pd.to_numeric(df[col], errors='coerce')
            s = s.dropna()
            if len(s) == 0:
                self._min[col] = 0.0
                self._max[col] = 1.0
                continue
            y = np.sort(s.values)
            n = len(y)
            a = int(round(n * self.rate))
            lo_idx = min(max(a, 0), n - 1)
            hi_idx = n - 1 if a == 0 else max(min(n - a - 1, n - 1), 0)
            self._min[col] = float(y[lo_idx])
            self._max[col] = float(y[hi_idx])
        return self

    def transform(self, df: pd.DataFrame):
        out = df.copy()
        for col in self._fitted_cols:
            if col in self.skip_cols:
                continue
            mn = self._min.get(col, 0.0)
            mx = self._max.get(col, 1.0)
            if mx == mn:
                continue
            out[col] = pd.to_numeric(out[col], errors='coerce')
            out.loc[out[col] > mx, col] = mx
            out.loc[out[col] < mn, col] = mn
            out[col] = (out[col] - mn) / (mx - mn)
        return out

    def fit_transform(self, df: pd.DataFrame, cols):
        return self.fit(df, cols).transform(df)


def eval_score(gt, pred):
    # 评估函数，计算均方误差
    mse = mean_squared_error(gt, pred)
    return mse


def get_col(data, args):
    col = data.columns

    # Reviewer-friendly: validate column count when using positional slicing
    if args.stage == 1:
        if len(col) < 155:
            raise ValueError(f"Stage 1 expects >=155 columns, got {len(col)}. "
                             "Please check your CSV schema or switch to name-based target.")
        feat_col = list(col)[5:155]
    elif args.stage == 2:
        if len(col) < 194:
            raise ValueError(f"Stage 2 expects >=194 columns, got {len(col)}. "
                             "Please check your CSV schema or switch to name-based target.")
        feat_col = list(col)[5:175] + list(col)[177:194]
    else:
        raise ValueError('Stage error: stage must be 1 or 2')

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
        raise ValueError('Target type error: targ must be int or str')

    print('Target:', target)
    targ_col = [target]

    if target not in data.columns:
        raise ValueError("Target col doesn't exist, check your input")

    return feat_col, targ_col



def make_composition_key(df):
    """Build a stable grouping key from composition columns (Element_*, Ratio_*).

    This is useful for GroupKFold to reduce composition-level leakage.
    """
    elem_cols = [c for c in df.columns if c.startswith('Element_')]
    ratio_cols = [c for c in df.columns if c.startswith('Ratio_')]
    # keep deterministic order
    elem_cols = sorted(elem_cols, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else x)
    ratio_cols = sorted(ratio_cols, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else x)

    # if ratio columns exist, use both; else only elements
    cols = elem_cols + ratio_cols
    if not cols:
        # fall back to index if no composition columns exist
        return pd.Series(np.arange(len(df)), name='group_key')

    # stringify to avoid float formatting issues; round ratios to a reasonable precision
    parts = []
    for c in cols:
        if c.startswith('Ratio_'):
            parts.append(df[c].astype(float).round(6).astype(str))
        else:
            parts.append(df[c].astype(str))
    key = parts[0]
    for p in parts[1:]:
        key = key + '|' + p
    return key


def get_data(args):
    data = pd.read_csv(args.train)

    feat_col, targ_col = get_col(data, args)

    groups = make_composition_key(data) if getattr(args, 'group_cv', False) else None

    train, val = split(
        data,
        num_split=args.num_split,
        shuffle=True,
        seed=getattr(args, 'seed', 2023),
        groups=groups
    )
    data = {'train': train, 'val': val}

    return data, feat_col, targ_col
