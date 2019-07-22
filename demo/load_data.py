import inspect
import os
import numpy as np
import pandas as pd
from config import configuration


class load_data(object):
    """
    数据加载类
    """

    def __init__(self):
        path = inspect.getfile(load_data)
        # demo_path
        self.demo_path = os.path.abspath(os.path.join(path, os.pardir))
        # ForecastScore_path
        self.ForecastScore_path = os.path.abspath('..')
        # data_path
        self.data = self.ForecastScore_path + '/data'

        # train_s1
        self.train_s1 = configuration.configuration().config_parameters("train_s1")
        # test_s1
        self.test_s1 = configuration.configuration().config_parameters("test_s1")
        # sample
        self.sample = configuration.configuration().config_parameters("sample")

    def get_train_s1(self, file_name, tag):
        """
        返回train_s1文件夹下的文件
        :param file_name: 文件名称
        :param tag: 返回的数据类型
        :return:
        """
        if tag == 'np':
            return np.loadtxt(self.data + self.train_s1[file_name + '_path'], delimiter=',', dtype=np.str)
        elif tag == 'pd':
            return pd.read_csv(self.data + self.train_s1[file_name + '_path'])
        else:
            return "请检查文件名/需要返回的数据类型"

    def get_test_s1(self, file_name, tag):
        """
        返回test_s1文件夹下的文件
        :param file_name: 文件名称
        :param tag: 返回的数据类型
        :return:
        """
        if tag == 'np':
            return np.loadtxt(self.data + self.test_s1[file_name + '_path'], delimiter=',', dtype=np.str)
        elif tag == 'pd':
            return pd.read_csv(self.data + self.test_s1[file_name + '_path'])
        else:
            return "请检查文件名/需要返回的数据类型"

    def get_sample(self, file_name, tag):
        """
        返回范例文件夹下的文件
        :param file_name: 文件名称
        :param tag: 返回的数据类型
        :return:
        """
        if tag == 'np':
            return np.loadtxt(self.data + self.sample[file_name + '_path'], delimiter=',', dtype=np.str)
        elif tag == 'pd':
            return pd.read_csv(self.data + self.sample[file_name + '_path'])
        else:
            return "请检查文件名/需要返回的数据类型"

# if __name__ == '__main__':
#
#     # 判断是否存在缺失值
#     test_s1_file_name = ['submission_s1']
#     train_s1_file_name = ['all_knowledge', 'course', 'course1_exams', 'course2_exams', 'course3_exams', 'course4_exams'
#                           , 'course5_exams', 'course6_exams', 'course7_exams', 'course8_exams', 'exam_score', 'student']
#     sample_s1_file_name = ['submission_s1_sample']
#
#     for file_name in test_s1_file_name:
#         df = load_data().get_test_s1(file_name, 'pd')
#         if (df.isnull().sum() == 0).all():
#             break
#
#     for file_name in train_s1_file_name:
#         df = load_data().get_train_s1(file_name, 'pd')
#         if (df.isnull().sum() == 0).all():
#             break
#
#     for file_name in sample_s1_file_name:
#         df = load_data().get_sample(file_name, 'pd')
#         if (df.isnull().sum() == 0).all():
#             break
