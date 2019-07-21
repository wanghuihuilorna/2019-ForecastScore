from demo.load_data import load_data


class preprocess(object):
    """
    数据预处理部分
    """

    def __init__(self):
        self.load_data = load_data()

    def get_trian_s1_student(self):
        """
        获取student.csv数据
        :return:
        """
        file_name = 'student'
        student = self.load_data.get_train_s1(file_name)
        student_id = student[1:, 0]
        gender = self.one_hot_coding(student[:, 1][1:]).tolist()

        return dict(zip(student_id, gender))

    def get_train_s1_course(self):
        """
        获取course.csv数据
        :return:
        """
        file_name = 'course'
        course = self.load_data.get_train_s1(file_name)
        course_id = course[:, 0]
        course_class = self.one_hot_coding(course[:, 1][1:]).tolist()

        return dict(zip(course_id, course_class))




# if __name__ == '__main__':
#     # a = [[1, 2, 3], [3, 4, 5]]
#     #
#     # tmp = preprocess().one_hot_coding(np.array(a[0]))
#     # print(tmp)
#
#     tmp = preprocess().get_trian_s1_student()
#     print(tmp)
#
#     tmp = preprocess().get_train_s1_course()
#     print(tmp)
