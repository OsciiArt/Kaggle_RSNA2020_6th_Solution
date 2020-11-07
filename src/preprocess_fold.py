import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, shutil, time, pickle, warnings, logging
import yaml
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics
from scipy.special import erfinv
from scipy.stats import mode

warnings.filterwarnings('ignore')


def data_split_StratifiedKFold(df, col_index, col_stratified, n_splits=5, random_state=42):
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
        np.arange(len(df)), y=df[col_stratified]))

    df_new = df[[col_index]]
    for fold in range(n_splits):
        df_new['fold{}_train'.format(fold + 1)] = 0
        df_new['fold{}_train'.format(fold + 1)][folds[fold][0]] = 1
        df_new['fold{}_valid'.format(fold + 1)] = 0
        df_new['fold{}_valid'.format(fold + 1)][folds[fold][1]] = 1

    return df_new


def data_split_KFold(df, col_index, n_splits=5, random_state=42):
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
        np.arange(len(df))))

    df_new = df[[col_index]]
    for fold in range(n_splits):
        df_new['fold{}_train'.format(fold + 1)] = 0
        df_new['fold{}_train'.format(fold + 1)][folds[fold][0]] = 1
        df_new['fold{}_valid'.format(fold + 1)] = 0
        df_new['fold{}_valid'.format(fold + 1)][folds[fold][1]] = 1

    return df_new


def data_split_GroupKFold(df, col_index, col_group, n_splits=5, random_state=42):
    """

    :param df:
    :param col_index:
    :param col_group:
    :param n_splits:
    :param random_state:
    :return:
    """
    group = np.sort(df[col_group].unique())
    print("num group: {}".format(len(group)))
    np.random.seed(random_state)
    group = group[np.random.permutation(len(group))]
    fold_list = []
    fold = 0
    count = 0
    fold_list.append([])
    for i, item in enumerate(group):
        count += (df[col_group] == item).sum()
        fold_list[fold].append(item)
        if count > len(df) / n_splits * (fold + 1):
            fold_list.append([])
            fold += 1

    df_new = df[[col_index]]
    for fold in range(n_splits):
        df_new['fold{}_train'.format(fold + 1)] = df[col_group].apply(
            lambda x: x not in fold_list[fold]).astype(np.int)
        df_new['fold{}_valid'.format(fold + 1)] = 1 - df_new['fold{}_train'.format(fold + 1)]

    for i in range(n_splits):
        print("fold: {}, valid: {}. group: {}".format(
            i + 1,
            (df_new['fold{}_valid'.format(i + 1)] == 1).sum(),
            len(fold_list[i]))
        )

    return df_new


def main():
    df = pd.read_csv("../input/melanoma/train.csv")


if __name__ == '__main__':
    main()