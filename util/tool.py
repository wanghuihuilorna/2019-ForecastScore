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
    """
    获取每个学生成绩的平均值
    :param df:
    :param params:
    :return:
    """
    # 仅仅取中值
    # mean = df.groupby('student_id').apply(lambda x: x['score'].mean())

    # 去除0值,去除最小值,最大值后取均值
    mean = df.groupby('student_id').apply(lambda x: list(x['score']))

    result = []
    for i in mean.values:
        if 0 in i:
            i.remove(0)
        i.remove(max(i))
        i.remove(min(i))
        result.append(np.array(i).mean())

    mean = pd.Series(result, index=mean.index)
    mean['0'] = result

    return mean


def get_median_value(df, **params):
    """
    获取每个学生成绩的中位数
    :param df:
    :param params:
    :return:
    """
    median = df.groupby('student_id').apply(lambda x: x['score'].median())

    return median


def get_mode_value(df, **params):
    """
    获取每个学生成绩的众数
    :param df:
    :param params:
    :return:
    """
    mode = df.groupby('student_id').apply(lambda x: x['score'].mode().max())

    return mode


def get_Maximum_value(df, **params):
    """
    获取每个学生成绩的最大值
    :param df:
    :param params:
    :return:
    """
    max = df.groupby('student_id').apply(lambda x: x['score'].max())

    return max


def get_Minimum_value(df, **params):
    """
    获取每个学生成绩的最小值
    :param df:
    :param params:
    :return:
    """
    min = df.groupby('student_id').apply(lambda x: x['score'].min())

    return min


def merge_all_knowledge(df, course_type='1'):
    """
    合并所有的知识点
    :param df:
    :param course_type:
    :return:
    """
    course_exam = preprocess.get_course_exams('course' + course_type + '_exams')
    all_knowledge = preprocess.get_all_knowledge('all_knowledge')
    all_knowledge = all_knowledge[all_knowledge.course == 'course' + course_type]

    course_exam.index = course_exam['exam_id'].tolist()
    del course_exam['exam_id']

    course_exam[course_exam.select_dtypes(include=['number']).columns] /= 100

    all_knowledge = all_knowledge[all_knowledge.course == 'course' + course_type]

    section = np.mat(all_knowledge.section).astype("float")
    category = np.mat(all_knowledge.category).astype("float")
    complexity = np.mat(all_knowledge.complexity).astype("float")
    course_exam = np.mat(course_exam.values).astype("float")

    section = np.dot(course_exam, section.T)
    category = np.dot(course_exam, category.T)
    complexity = np.dot(course_exam, complexity.T)

    df['section'] = section.tolist()
    df['category'] = category.tolist()
    df['complexity'] = complexity.tolist()

    return df
