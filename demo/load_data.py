import inspect
import os
import numpy as np
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

    def get_train_s1(self, file_name):
        """
        返回train_s1文件夹下的文件
        :return:
        """
        return np.loadtxt(self.data + self.train_s1[file_name + '_path'], delimiter=',', dtype=np.str)

    def get_test_s1(self, file_name):
        """
        返回test_s1文件夹下的文件
        :param file_name:
        :return:
        """
        return np.loadtxt(self.data + self.test_s1[file_name + '_path'], delimiter=',', dtype=np.str)

    def get_sample(self, file_name):
        """
        返回范例文件夹下的文件
        :param file_name:
        :return:
        """
        return np.loadtxt(self.data + self.sample[file_name + '_path'], delimiter=',', dtype=np.str)


# if __name__ == '__main__':
#     file_name = 'all_knowledge'
#
#     print(load_data().get_train_s1(file_name))
