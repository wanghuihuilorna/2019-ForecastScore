import numpy as np
import pandas as pd
import pickle
from demo import preprocess
from sklearn.preprocessing import MinMaxScaler
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


def label_encoding(df, columns: list = None, **params):
    """
    标签编码
    :param df:
    :param params:
    :return:
    """
    if columns is None:
        columns = df.columns

    le = LabelEncoder()
    for feature in columns:
        df[feature] = le.fit_transform(df[feature])

    return df


def missing_value_padding(df, columns=None, **params):
    """
    缺失值填充
    :param df:
    :param params:
    :return:
    """
    if columns is None:
        columns = df.columns

    for feature in columns:
        df[feature] = pd.Series([df[feature][c].value_counts().index[0] if df[feature][c].dtype == np.dtype('O') else
                                 df[feature][c].median() for c in df[feature]],  # 计算中位数
                                index=df[feature].columns)


def min_max_scaler(df, **params):
    """
    归一化
    :param df:
    :param params:
    :return:
    """
    df = MinMaxScaler().fit_transform(df)

    return df


def save_model(model, model_path):
    """
    保存模型
    :param model:
    :param model_path:
    :return:
    """
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def load_model(model_path):
    """
    加载模型
    :param model_path:
    :return:
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model


def get_mean_value(df, **params):
    pass


# if __name__ == "__main__":
#     tmp = preprocess.get_exam_score('exam_score', ['1'])
#     train_X = tmp[tmp.exam_id.isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])]
#     mean_value = train_X.groupby('advert_id').apply(lambda x: x[['advert_showed']].cumsum())
#
#     print(train_X.head())
