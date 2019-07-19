# -*- coding: utf-8 -*-

# 注 python2为ConfigParser python3为configparser
from configparser import ConfigParser

import os
import inspect


class configuration(object):
    """
    配置操作类
    """

    def __init__(self):
        # 初始化类 注意路径问题
        self.config_parser = ConfigParser()
        self.path = os.path.abspath(os.path.join(inspect.getfile(configuration), os.pardir))

    def config_parameters(self, custom_parameters):
        """
        :return: 字典形式返回参数值
        """
        self.config_parser.read(self.path + '/config.ini')

        # 得到该section的所有键值对
        custom_parameters_section = self.config_parser.items(custom_parameters)

        # 字典的形式返回结果
        return dict(zip([i[0] for i in custom_parameters_section], [i[1] for i in custom_parameters_section]))

# if __name__ == '__main__':
#     tes1_s1 = configuration().config_parameters('test_s1')
#     train_s1 = configuration().config_parameters('train_s1')
#     sample = configuration().config_parameters('sample')
#
#     print(tes1_s1)
#     print(train_s1)
#     print(sample)
