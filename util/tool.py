import os
import numpy as np
import pandas as pd
import pickle
import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from util.load_data import load_data


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


def get_maximum_value(df, **params):
    """
    获取每个学生成绩的最大值
    :param df:
    :param params:
    :return:
    """
    max = df.groupby('student_id').apply(lambda x: x['score'].max())

    return max


def get_minimum_value(df, **params):
    """
    获取每个学生成绩的最小值
    :param df:
    :param params:
    :return:
    """
    min = df.groupby('student_id').apply(lambda x: x['score'].min())

    return min


def reduce_mem_usage(df, verbose=True):
    """
    减少内存消耗
    :param df:
    :param verbose:
    :return:
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def get_course(filename, tag='pd'):
    """
    处理course.csv文件
    :param filename:
    :return:
    """
    df = load_data().get_train_s1(filename, tag)
    df = label_encoding(df, columns=[u'course_class'])

    return df


def get_student(filename, tag='pd'):
    """
    处理student.csv文件
    :param filename:
    :return:
    """
    df = load_data().get_train_s1(filename, tag)

    return df


def get_all_knowledge(filename, tag='pd'):
    """
    处理all_knowledge.csv文件
    :param filename:
    :return:
    """
    df = load_data().get_train_s1(filename, tag)

    # v1.0需要
    # for feature in ['section', 'category']:
    #     df[feature] = [x.split(':')[-1] for x in df[feature]]

    return df


def get_course_exams(filename, tag='pd'):
    """
    处理course1_exams.csv-course8_exams.csv
    :param filename:
    :return:
    """
    df = load_data().get_train_s1(filename, tag)

    return df


def merge_all_knowledge(df, course_type=None):
    """
    合并all_knowledge数据
    :param df:
    :param course_type:
    :return:
    """
    # 获取course_exams
    course_exams = get_course_exams(course_type)

    # 获取 知识点信息
    all_knowledge = get_all_knowledge('all_knowledge')
    # 获取 course_id对应的知识点信息
    # all_knowledge = all_knowledge[all_knowledge.course == course_type.split('/')[-1].split('_')[0]]
    all_knowledge = all_knowledge[all_knowledge.course == course_type.split('_')[0]]

    # 构造 columns
    columns = course_exams.columns.to_list()
    columns.extend(sorted(list(set(all_knowledge['section'])), key=lambda x: int(x.split(':')[-1])))
    columns.extend(sorted(list(set(all_knowledge['category'])), key=lambda x: int(x.split(':')[-1])))
    columns.extend(
        sorted(['Y:' + str(i) for i in list(set(all_knowledge['complexity']))], key=lambda x: int(x.split(':')[-1])))

    # 更换 columns
    course_exams = course_exams.reindex(columns=columns)
    # 填充 0
    course_exams.fillna(0, inplace=True)

    for i in range(all_knowledge.shape[0]):
        # 获取 一行数据
        line = all_knowledge.iloc[i]
        # 获取 knowledge_point
        knowledge_point = course_exams[line['knowledge_point']]
        # 获取 section
        course_exams[line['section']] += knowledge_point
        # 获取 category
        course_exams[line['category']] += knowledge_point
        # 获取 complexity
        course_exams['Y:' + str(line['complexity'])] += knowledge_point

    # 提取/合并 K：i 知识点
    exam_score = pd.merge(df, course_exams, how='left', on='exam_id')

    return exam_score


def create_score_feature(df, tag='pd'):
    """
    提取特征：
    """
    features = []
    for i, row in tqdm(df.iterrows()):
        m = [int(i) for i in row['score'] if int(i) >= 50]
        features.append([np.mean(m), np.median(m), np.std(m), np.max(m), np.min(m), np.mean(m[-2:]), np.std(m[-2:])])

    feats = pd.DataFrame(features)
    feats.columns = ['features{}'.format(i) for i in range(feats.shape[1])]
    return feats


def get_exam_score(filename, course_id: str, tag='pd', save=True):
    """
    处理exam_score.csv  course_id 只要一个值 如： 'course1'
    :param filename:
    :param course_id:
    :param tag:
    :param save:
    :return:
    """
    # .h5文件保存路径
    save_path = load_data().get_project_path() + '/data/cache/%s_%s.h5' % (filename, course_id)

    if os.path.exists(save_path):
        exam_score = reduce_mem_usage(pd.read_hdf(path_or_buf=save_path, mode='r', key=course_id))
    else:
        exam_score = load_data().get_train_s1(filename, tag)

        # 获取 学生信息
        student = get_student('student')
        # 获取 课程信息
        course = get_course('course')

        # 提取/合并 性别信息
        exam_score = pd.merge(exam_score, student, how='left', on='student_id')
        # 提取/合并 course_class信息
        exam_score = pd.merge(exam_score, course, how='left', on='course')

        # 获取选中的 course_id
        exam_score = exam_score[exam_score['course'] == course_id]

        # 合并 section/category/complexity
        exam_score = merge_all_knowledge(exam_score, course_id + '_exams')

        # 读取特定的course_exams.csv文件
        course_exams = get_course_exams(course_id + '_exams')

        # 处理exam_id
        exam_score['exam_id'] = exam_score['exam_id'].map(
            lambda x: dict(zip(course_exams['exam_id'], [i for i in range(len(course_exams['exam_id']))]))[x])

        # 取均值
        mean_value = get_mean_value(exam_score)

        # 使用均值来填充空值
        result = None
        for tmp in exam_score.groupby('student_id'):
            tmp[1]['score'].replace(0, mean_value[tmp[0]], inplace=True)
            if result is None:
                result = tmp[1]
            else:
                result = pd.concat([result, tmp[1]], axis=0)

        result.reset_index(drop=True, inplace=True)

        exam_score = result

        # log1p就是log(1+x)，用来对得分进行数据预处理，它的好处是转化后的数据更加服从高斯分布，有利于后续的分类结果。
        # 需要注意，最后需要将预测出的平滑数据还原，而还原过程就是log1p的逆运算expm1。
        exam_score["score"] = np.log1p(exam_score["score"])

        # 删除列值都相同的列
        exam_score = exam_score.ix[:, (exam_score != exam_score.ix[0]).any()]

        # 保存数据
        if save is True:
            exam_score.to_hdf(path_or_buf=save_path, key=course_id)

    return exam_score


def get_submission_s1(filename, course_id: str, tag='pd', save=True):
    """
    处理submission_s1.csv文件
    :param filename:
    :param course_id:
    :param tag:
    :return:
    """
    # .h5文件保存路径
    save_path = load_data().get_project_path() + '/data/cache/%s_%s.h5' % (filename, course_id)

    if os.path.exists(save_path):
        submission_s1 = reduce_mem_usage(pd.read_hdf(path_or_buf=save_path, mode='r', key=course_id))
    else:
        submission_s1 = load_data().get_test_s1(filename, tag)

        # 获取 学生信息
        student = get_student('student')
        # 获取 课程信息
        course = get_course('course')

        # 提取/合并 性别信息
        submission_s1 = pd.merge(submission_s1, student, how='left', on='student_id')
        # 提取/合并 course_class信息
        submission_s1 = pd.merge(submission_s1, course, how='left', on='course')

        # 获取选中的 course_id
        submission_s1 = submission_s1[submission_s1['course'] == course_id]

        # 合并 section/category/complexity
        submission_s1 = merge_all_knowledge(submission_s1, course_id + '_exams')

        # 读取特定的course_exams.csv文件
        course_exams = get_course_exams(course_id + '_exams')

        # 处理exam_id
        submission_s1['exam_id'] = submission_s1['exam_id'].map(
            lambda x: dict(zip(course_exams['exam_id'], [i for i in range(len(course_exams['exam_id']))]))[x])

        submission_s1['pred'] = 0

        # 删除列值都相同的列
        submission_s1 = submission_s1.ix[:, (submission_s1 != submission_s1.ix[0]).any()]

        # 保存数据
        if save is True:
            submission_s1.to_hdf(path_or_buf=save_path, key=course_id)

    return submission_s1

######################################################## v1.0 ########################################################
# def merge_all_knowledge(df, course_type=None):
#     """
#     合并all_knowledge数据
#     :param df:
#     :param course_type:
#     :return:
#     """
#     course_exam = get_course_exams(course_type)
#     del course_exam['exam_id']
#
#     to_calculate_columns = df[course_exam.columns.to_list()]
#
#     all_knowledge = get_all_knowledge('all_knowledge')
#     all_knowledge = all_knowledge[all_knowledge.course == course_type.split('_')[0]]
#
#     to_calculate_columns[to_calculate_columns.select_dtypes(include=['number']).columns] /= 100
#
#     all_knowledge = all_knowledge[all_knowledge.course == course_type.split('_')[0]]
#
#     section = np.mat(all_knowledge.section).astype("float")
#     category = np.mat(all_knowledge.category).astype("float")
#     complexity = np.mat(all_knowledge.complexity).astype("float")
#     to_calculate_columns = np.mat(to_calculate_columns.values).astype("float")
#
#     section = np.dot(to_calculate_columns, section.T)
#     category = np.dot(to_calculate_columns, category.T)
#     complexity = np.dot(to_calculate_columns, complexity.T)
#
#     df['section'] = section.reshape(1, -1).tolist()[0]
#     df['category'] = category.reshape(1, -1).tolist()[0]
#     df['complexity'] = complexity.reshape(1, -1).tolist()[0]
#
#     return df
#
# def get_exam_score(filename, course_id: str, tag='pd', save=True):
#     """
#     处理exam_score.csv  course_id 只要一个值
#     :param filename:
#     :return:
#     """
#     # .h5文件保存路径
#     save_path = load_data().get_project_path() + '/data/cache/%s_%s.h5' % (filename, course_id)
#
#     if os.path.exists(save_path):
#         exam_score = reduce_mem_usage(pd.read_hdf(path_or_buf=save_path, mode='r', key=course_id))
#     else:
#         exam_score = load_data().get_train_s1(filename, tag)
#
#         student = get_student('student')
#         course = get_course('course')
#
#         # 合并性别
#         exam_score = pd.merge(exam_score, student, how='left', on='student_id')
#         # 合并course_class
#         exam_score = pd.merge(exam_score, course, how='left', on='course')
#
#         # 获取选中的course_id
#         exam_score = exam_score[exam_score['course'] == course_id]
#
#         # 读取特定的course_exams.csv文件
#         course_exams = get_course_exams(course_id + '_exams')
#         # 合并course_exams
#         exam_score = pd.merge(exam_score, course_exams, how='left', on='exam_id')
#         # 合并section/category/complexity
#         exam_score = merge_all_knowledge(exam_score, course_id + '_exams')
#
#         # 处理标签数据
#         exam_score = label_encoding(exam_score, ['course'])
#
#         # 处理exam_id
#         exam_score['exam_id'] = exam_score['exam_id'].map(
#             lambda x: dict(zip(course_exams['exam_id'], [i for i in range(len(course_exams['exam_id']))]))[x])
#
#         # 取均值
#         mean_value = get_mean_value(exam_score)
#
#         # 使用均值来填充空值
#         result = None
#         for tmp in exam_score.groupby('student_id'):
#             tmp[1]['score'].replace(0, mean_value[tmp[0]], inplace=True)
#             if result is None:
#                 result = tmp[1]
#             else:
#                 result = pd.concat([result, tmp[1]], axis=0)
#
#         result.reset_index(drop=True, inplace=True)
#
#         exam_score = result
#
#         # log1p就是log(1+x)，用来对得分进行数据预处理，它的好处是转化后的数据更加服从高斯分布，有利于后续的分类结果。
#         # 需要注意，最后需要将预测出的平滑数据还原，而还原过程就是log1p的逆运算expm1。
#         exam_score["score"] = np.log1p(exam_score["score"])
#
#         ####################################     是否要转str    #######################################
#         # exam_score['student_id'] = exam_score['student_id'].astype(str)
#         # # 使用.get_dummies()方法对特征矩阵进行类似“坐标投影”操作。获得在新空间下的特征表达。
#         # exam_score['student_id'] = pd.get_dummies(exam_score['student_id']).reset_index(drop=True)
#
#         # exam_score['course'] = exam_score['course'].astype(str)
#         # exam_score['gender'] = exam_score['gender'].astype(str)
#         # exam_score['course_class'] = exam_score['course_class'].astype(str)
#         ####################################     是否要转str    #######################################
#
#         # 删除全0的列
#         # exam_score = exam_score.ix[:, ~((exam_score == 0).all())]
#
#         # 保存数据
#         if save is True:
#             exam_score.to_hdf(path_or_buf=save_path, key=course_id)
#
#     return exam_score
#
#
# def get_submission_s1(filename, course_id: str, tag='pd', save=True):
#     """
#     处理submission_s1.csv文件
#     :param filename:
#     :param course_id:
#     :param tag:
#     :return:
#     """
#     # .h5文件保存路径
#     save_path = load_data().get_project_path() + '/data/cache/%s_%s.h5' % (filename, course_id)
#
#     if os.path.exists(save_path):
#         submission_s1 = reduce_mem_usage(pd.read_hdf(path_or_buf=save_path, mode='r', key=course_id))
#     else:
#         submission_s1 = load_data().get_test_s1(filename, tag)
#
#         # 0值填充
#         submission_s1.fillna(0, inplace=True)
#         student = get_student('student')
#         course = get_course('course')
#
#         # 合并性别
#         submission_s1 = pd.merge(submission_s1, student, how='left', on='student_id')
#         # 合并course_class
#         submission_s1 = pd.merge(submission_s1, course, how='left', on='course')
#
#         if len(course_id) != 0:
#             # 获取选中的course_id
#             submission_s1 = submission_s1[submission_s1['course'] == course_id]
#             # 使用knowledge_point 替换 exam_id
#             course_id_exams = course_id + '_exams'
#             # 读取特定的course_exams.csv文件
#             course_exams = get_course_exams(course_id_exams)
#             # 合并
#             submission_s1 = pd.merge(submission_s1, course_exams, how='left', on='exam_id')
#             submission_s1 = merge_all_knowledge(submission_s1, course_id_exams)
#
#             # 处理标签特征
#             # submission_s1.index = list(submission_s1['student_id'])
#             # del submission_s1['student_id']
#             submission_s1 = label_encoding(submission_s1, ['course', 'exam_id'])
#
#             # # 删除全0的列
#             # submission_s1 = submission_s1.ix[:, ~((submission_s1 == 0).all())]
#             #
#             # submission_s1['pred'] = 0
#
#             # 保存数据
#             if save is True:
#                 submission_s1.to_hdf(path_or_buf=save_path, key=course_id)
#
#     return submission_s1
