import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def one_hot_coding(df, **params):
    """
    one-hot编码
    :param :
    :return:
    """
    oh = OneHotEncoder()
    for feature in df.columns:
        df[feature] = oh.fit_transform(df[feature])

    return df


def label_encoding(df, **params):
    """
    标签编码
    :param df:
    :param params:
    :return:
    """
    le = LabelEncoder()
    for feature in df.columns:
        df[feature] = le.fit_transform(df[feature])

    return df


def missing_value_padding(df, **params):
    """
    缺失值填充
    :param df:
    :param params:
    :return:
    """
    for feature in df.columns:
        df[feature] = pd.Series([df[feature][c].value_counts().index[0] if df[feature][c].dtype == np.dtype('O') else
                                 df[feature][c].median() for c in df[feature]],  # 计算中位数
                                index=df[feature].columns)
