import pandas as pd
import time
from demo.preprocess import *


# class statisticalAnalysis(object):
#     def __init__(self):
#
#     def main(self):


if __name__ == '__main__':
    start = time.clock()

    # course = get_course('course')
    #
    # student = get_student('student')
    #
    # all_knowledge = get_all_knowledge('all_knowledge')
    #
    # tmp = []
    # for file_name in ['course1_exams', 'course2_exams', 'course3_exams', 'course4_exams'
    #     , 'course5_exams', 'course6_exams', 'course7_exams', 'course8_exams']:
    #     tmp.extend(get_course_exams(file_name))
    #
    # tmp = get_exam_score('exam_score', ['1'])
    # train_X = tmp[tmp.exam_id.isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])]
    # test_X = tmp[tmp.exam_id.isin([16, 17])]
    #
    # train_y = train_X.score
    # test_y = test_X.score
    #
    # mean_value = tool.get_mean_value(train_X)
    # median_value = tool.get_median_value(train_X)
    # mode_value = tool.get_mode_value(train_X)
    # max_value = tool.get_Maximum_value(train_X)
    # min_value = tool.get_Minimum_value(train_X)
    #
    # mean_median_value = (mean_value + median_value) / 2
    #
    # del train_X['score']
    # del test_X['score']
    #
    # for value in [mean_median_value, mean_value, median_value, mode_value, max_value, min_value]:
    #     value = value.to_frame()
    #     predictions = pd.merge(test_X, value, how='left', on='student_id')[0]
    #
    #     test_y = np.array(test_y)
    #     rmse = np.sqrt(((test_y - predictions) ** 2).mean())
    #     print(rmse)
    #
    # print(time.clock() - start)



