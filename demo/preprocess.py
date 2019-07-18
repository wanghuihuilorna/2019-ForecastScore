import numpy as np
from sklearn import preprocessing
from demo.load_data import load_data


class preprocess(object):
    """
    数据预处理部分
    """

    def __init__(self):
        self.load_data = load_data()

    def one_hot_coding(self, raw_vector):
        """
        one-hot编码
        :param raw_vector: 行向量
        :return:
        """
        # 列向量
        column_vector = raw_vector.reshape(len(raw_vector), -1)

        # one_hot 编码函数
        result = preprocessing.OneHotEncoder(categories='auto').fit_transform(column_vector).toarray()

        return result

    def get_trian_s1_student(self):
        """
        获取student.csv数据
        :return:
        """
        file_name = 'student'
        student = self.load_data.get_train_s1(file_name)
        student_id = student[:, 0]
        gender = self.one_hot_coding(student[:, 1][1:])
        # 构造列名
        gender_name = np.array([['gender_'+str(i)] for i in range(gender.shape[1])])
        # 重新生成gender
        gender = np.concatenate((gender_name.reshape(-1, gender.shape[1]), gender), axis=0)

        return np.concatenate((student_id.reshape(-1, 1), gender), axis=1)


# if __name__ == '__main__':
#     a = [[1, 2, 3], [3, 4, 5]]
#
#     tmp = preprocess().one_hot_coding(np.array(a[0]))
#     print(tmp)

    # tmp = preprocess().get_trian_s1_student()
    # print(tmp)
