import scipy.sparse as ss
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from util.load_data import load_data
from util import tool
from sklearn.svm import SVC
import time
import xgboost as xgb


def get_course(filename, tag='pd'):
    """
    处理course.csv文件
    :param filename:
    :return:
    """
    df = load_data().get_train_s1(filename, tag)
    df = tool.label_encoding(df, columns=[u'course_class'])

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

    for feature in ['section', 'category']:
        df[feature] = [x.split(':')[-1] for x in df[feature]]

    return df


def get_course_exams(filename, tag='pd'):
    """
    处理course1_exams.csv-course8_exams.csv
    :param filename:
    :return:
    """
    df = load_data().get_train_s1(filename, tag)

    return df


def xgb_model(model_name, train_X, train_y, test_X, test_y=None):
    """
    xgb模型
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    """
    gbm = xgb.XGBRegressor(max_depth=10, n_estimators=1000, learning_rate=0.01).fit(train_X, train_y)

    model_path = load_data().get_project_path() + '/model/' + model_name
    tool.save_model(gbm, model_path=model_path)

    if test_X is None:
        return None

    predictions = gbm.predict(test_X)

    if test_y is not None:
        test_y = np.array(test_y)
        predictions = np.array(predictions)
        rmse = np.sqrt(((test_y - predictions) ** 2).mean())

        return rmse, predictions
    else:
        return predictions


def lgb_model(model_name, train_X, train_y, test_X, test_y=None):
    """
    lgb模型
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    """
    gbm = lgb.LGBMRegressor(max_depth=10, n_estimators=1000, learning_rate=0.01).fit(train_X, train_y)

    model_path = load_data().get_project_path() + '/model/' + model_name
    tool.save_model(gbm, model_path=model_path)

    if test_X is None:
        return None

    predictions = gbm.predict(test_X)

    if test_y is not None:
        test_y = np.array(test_y)
        predictions = np.array(predictions)
        rmse = np.sqrt(((test_y - predictions) ** 2).mean())

        return rmse, predictions
    else:
        return predictions


def svm_model(model_name, train_X, train_y, test_X, test_y=None):
    """
    svm模型
    :param model_name:
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    """
    svm_model = SVC(kernel='rbf', gamma='auto').fit(train_X, train_y)
    model_path = load_data().get_project_path() + '/model/' + model_name
    tool.save_model(svm_model, model_path=model_path)

    if test_X is None:
        return None

    predictions = svm_model.predict(test_X)

    if test_y is not None:
        test_y = np.array(test_y)
        predictions = np.array(predictions)
        rmse = np.sqrt(((test_y - predictions) ** 2).mean())

        return rmse, predictions
    else:
        return predictions


def merge_all_knowledge(df, course_type=None):
    """
    合并all_knowledge数据
    :param df:
    :param course_type:
    :return:
    """
    course_exam = get_course_exams(course_type)
    del course_exam['exam_id']

    to_calculate_columns = df[course_exam.columns.to_list()]

    all_knowledge = get_all_knowledge('all_knowledge')
    all_knowledge = all_knowledge[all_knowledge.course == course_type.split('_')[0]]

    to_calculate_columns[to_calculate_columns.select_dtypes(include=['number']).columns] /= 100

    all_knowledge = all_knowledge[all_knowledge.course == course_type.split('_')[0]]

    section = np.mat(all_knowledge.section).astype("float")
    category = np.mat(all_knowledge.category).astype("float")
    complexity = np.mat(all_knowledge.complexity).astype("float")
    to_calculate_columns = np.mat(to_calculate_columns.values).astype("float")

    section = np.dot(to_calculate_columns, section.T)
    category = np.dot(to_calculate_columns, category.T)
    complexity = np.dot(to_calculate_columns, complexity.T)

    df['section'] = section.reshape(1, -1).tolist()[0]
    df['category'] = category.reshape(1, -1).tolist()[0]
    df['complexity'] = complexity.reshape(1, -1).tolist()[0]

    return df


def get_exam_score(filename, course_id: str, tag='pd', save=True):
    """
    处理exam_score.csv  course_id 只要一个值
    :param filename:
    :return:
    """
    # .h5文件保存路径
    save_path = load_data().get_project_path() + '/data/cache/%s_%s.h5' % (filename, course_id)

    if os.path.exists(save_path):
        exam_score = tool.reduce_mem_usage(pd.read_hdf(path_or_buf=save_path, mode='r', key=course_id))
    else:
        exam_score = load_data().get_train_s1(filename, tag)

        student = get_student('student')
        course = get_course('course')

        # 合并性别
        exam_score = pd.merge(exam_score, student, how='left', on='student_id')
        # 合并course_class
        exam_score = pd.merge(exam_score, course, how='left', on='course')

        # 获取选中的course_id
        exam_score = exam_score[exam_score['course'] == course_id]

        # 读取特定的course_exams.csv文件
        course_exams = get_course_exams(course_id + '_exams')
        # 合并course_exams
        exam_score = pd.merge(exam_score, course_exams, how='left', on='exam_id')
        # 合并section/category/complexity
        exam_score = merge_all_knowledge(exam_score, course_id + '_exams')

        # 处理标签数据
        exam_score = tool.label_encoding(exam_score, ['course'])

        # 处理exam_id
        exam_score['exam_id'] = exam_score['exam_id'].map(
            lambda x: dict(zip(course_exams['exam_id'], [i for i in range(len(course_exams['exam_id']))]))[x])

        # 取均值
        mean_value = tool.get_mean_value(exam_score)

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

        # 保存数据
        if save is True:
            exam_score.to_hdf(path_or_buf=save_path, key=course_id)

    return exam_score


def get_submission_s1(filename, course_id: list, tag='pd'):
    """
    处理submission_s1.csv文件
    :param filename:
    :param course_id:
    :param tag:
    :return:
    """
    submission_s1 = load_data().get_test_s1(filename, tag)

    # 0值填充
    submission_s1.fillna(0, inplace=True)
    student = get_student('student')
    course = get_course('course')

    # 合并性别
    submission_s1 = pd.merge(submission_s1, student, how='left', on='student_id')
    # 合并course_class
    submission_s1 = pd.merge(submission_s1, course, how='left', on='course')

    if len(course_id) != 0:
        course_list = ['course' + i for i in course_id]
        # 获取选中的course_id
        submission_s1 = submission_s1[submission_s1['course'].isin(course_list)]
        # 使用knowledge_point 替换 exam_id
        course_list = [i + '_exams' for i in course_list]

        for course_i in course_list:
            # 读取特定的course_exams.csv文件
            course_exams = get_course_exams(course_i)
            # 合并
            submission_s1 = pd.merge(submission_s1, course_exams, how='left', on='exam_id')
            submission_s1 = merge_all_knowledge(submission_s1, course_i)

        # 处理标签特征
        # submission_s1.index = list(submission_s1['student_id'])
        # del submission_s1['student_id']
        submission_s1 = tool.label_encoding(submission_s1, ['course', 'exam_id'])

        return submission_s1
    else:
        return submission_s1


if __name__ == '__main__':
    start = time.clock()
    # # 判断是否存在缺失值
    # test_s1_file_name = ['submission_s1']
    # train_s1_file_name = ['all_knowledge', 'course', 'course1_exams', 'course2_exams', 'course3_exams', 'course4_exams'
    #     , 'course5_exams', 'course6_exams', 'course7_exams', 'course8_exams', 'exam_score', 'student']
    # sample_s1_file_name = ['submission_s1_sample']
    #
    #########################################################xgb测试#################################################################
    # # 获取course
    # course = get_course('course')
    # # 获取student
    # student = get_student('student')
    # # 获取all_knowledge
    # all_knowledge = get_all_knowledge('all_knowledge')
    #
    # for course_id in ['course1', 'course2', 'course3', 'course4', 'course5', 'course6', 'course7', 'course8']:
    #     # 获取exam_score
    #     exam_score = get_exam_score('exam_score', course_id)
    #     # 获取exam_id
    #     exam_id = list(set(exam_score.exam_id))
    #     # 对exam_id进行排序
    #     exam_id.sort()
    #     # 获取训练集
    #     train_X = exam_score[exam_score.exam_id.isin(exam_id[:int(len(exam_id) - 2)])]
    #     # 获取测试集
    #     test_X = exam_score[exam_score.exam_id.isin(exam_id[int(len(exam_id) - 2):])]
    #
    #     # 得分
    #     train_y = train_X.score
    #     # 得分
    #     test_y = test_X.score
    #
    #     del train_X['score']
    #     del test_X['score']
    #
    #     rmse, predictions = xgb_model('xgb_model_1.pkl', train_X, train_y, test_X, test_y)
    #     print(rmse)

    #############################################################lgb测试#################################################################
    # # 获取course
    # course = get_course('course')
    # # 获取student
    # student = get_student('student')
    # # 获取all_knowledge
    # all_knowledge = get_all_knowledge('all_knowledge')
    #
    # for course_id in ['course1', 'course2', 'course3', 'course4', 'course5', 'course6', 'course7', 'course8']:
    #     # 获取exam_score
    #     exam_score = get_exam_score('exam_score', course_id)
    #     # 获取exam_id
    #     exam_id = list(set(exam_score.exam_id))
    #     # 对exam_id进行排序
    #     exam_id.sort()
    #     # 获取训练集
    #     train_X = exam_score[exam_score.exam_id.isin(exam_id[:int(len(exam_id) - 2)])]
    #     # 获取测试集
    #     test_X = exam_score[exam_score.exam_id.isin(exam_id[int(len(exam_id) - 2):])]
    #
    #     # 得分
    #     train_y = train_X.score
    #     # 得分
    #     test_y = test_X.score
    #
    #     del train_X['score']
    #     del test_X['score']
    #
    #     rmse, predictions = lgb_model('lgb_model_1.pkl', train_X, train_y, test_X, test_y)
    #     print(rmse)
    ####################################################svm测试#################################################################
    # # 获取course
    # course = get_course('course')
    # # 获取student
    # student = get_student('student')
    # # 获取all_knowledge
    # all_knowledge = get_all_knowledge('all_knowledge')
    #
    # for course_id in ['course1', 'course2', 'course3', 'course4', 'course5', 'course6', 'course7', 'course8']:
    #     # 获取exam_score
    #     exam_score = get_exam_score('exam_score', course_id)
    #     # 获取exam_id
    #     exam_id = list(set(exam_score.exam_id))
    #     # 对exam_id进行排序
    #     exam_id.sort()
    #     # 获取训练集
    #     train_X = exam_score[exam_score.exam_id.isin(exam_id[:int(len(exam_id) - 2)])]
    #     # 获取测试集
    #     test_X = exam_score[exam_score.exam_id.isin(exam_id[int(len(exam_id) - 2):])]
    #
    #     # 得分
    #     train_y = train_X.score
    #     # 得分
    #     test_y = test_X.score
    #
    #     del train_X['score']
    #     del test_X['score']
    #
    #     rmse, predictions = svm_model('svm_model_1.pkl', train_X.astype('float'), train_y.astype('float'), test_X.astype('float'), test_y.astype('float'))
    #     print(rmse)
    ####################################################测试#################################################################

    ####################################################xgb#################################################################
    reuslt = []
    for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
        submission_s1 = get_submission_s1('submission_s1', [i])

        del submission_s1['pred']

        exam_score = get_exam_score('exam_score', 'course' + i)

        train_y = exam_score['score']

        del exam_score['score']

        predictions = xgb_model(model_name='xgb_model_' + i + '.pkl', train_X=exam_score, train_y=train_y,
                                test_X=submission_s1, test_y=None)

        reuslt.extend(predictions.tolist())

    submit = load_data().get_test_s1('submission_s1', 'pd')
    submit['pred'] = reuslt
    submit.to_csv(load_data().get_project_path() + '/data/test_s1/submission_s1_sample_xgb.csv', index=None, encoding='utf-8')
    ####################################################lgb#################################################################
    # reuslt = []
    # for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
    #     submission_s1 = get_submission_s1('submission_s1', [i])
    #
    #     del submission_s1['pred']
    #
    #     exam_score = get_exam_score('exam_score', [i])
    #
    #     train_y = exam_score['score']
    #
    #     del exam_score['score']
    #
    #     predictions = lgb_model(model_name='lgb_model_' + i + '.pkl', train_X=exam_score, train_y=train_y,
    #                             test_X=submission_s1, test_y=None)
    #
    #     reuslt.extend(predictions.tolist())
    #
    # submit = load_data().get_test_s1('submission_s1', 'pd')
    # submit['pred'] = reuslt
    # submit.to_csv(load_data().get_project_path() + '/data/test_s1/submission_s1_sample_lgb.csv', index=None, encoding='utf-8')
    ####################################################svm#################################################################
    # reuslt = []
    # for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
    #     print(i)
    #     submission_s1 = get_submission_s1('submission_s1', [i])
    #
    #     del submission_s1['pred']
    #
    #     exam_score = get_exam_score('exam_score', [i])
    #
    #     train_y = exam_score['score']
    #
    #     del exam_score['score']
    #
    #     predictions = svm_model(model_name='svm_model_' + i + '.pkl', train_X=exam_score, train_y=train_y,
    #                             test_X=submission_s1, test_y=None)
    #
    #     reuslt.extend(predictions.tolist())
    #
    # submit = load_data().get_test_s1('submission_s1', 'pd')
    # submit['pred'] = reuslt
    # submit.to_csv(load_data().get_project_path() + '/data/test_s1/submission_s1_sample_svm.csv', index=None, encoding='utf-8')

    ####################################################mean-median#################################################################
    # reuslt = []
    # for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
    #     print(i)
    #     train_X = get_exam_score('exam_score', [i])
    #     test_X = get_submission_s1('submission_s1', [i])
    #     del test_X['pred']
    #
    #     mean_value = tool.get_mean_value(train_X)
    #     median_value = tool.get_median_value(train_X)
    #     # mode_value = tool.get_mode_value(train_X)
    #     # max_value = tool.get_Maximum_value(train_X)
    #     # min_value = tool.get_Minimum_value(train_X)
    #
    #     mean_median_value = (mean_value + median_value) / 2
    #
    #     mean_median_value = mean_median_value.to_frame()
    #     predictions = pd.merge(test_X, mean_median_value, how='left', on='student_id')[0]
    #
    #     reuslt.extend(predictions.tolist())
    #
    # submit = load_data().get_test_s1('submission_s1', 'pd')
    # submit['pred'] = reuslt
    # submit.to_csv(load_data().get_project_path() + '/data/test_s1/submission_s1_sample_mean_median.csv', index=None,
    #               encoding='utf-8')

    ########################################################mean-median-xgb#################################################################
    # data1 = load_data().get_test_s1('submission_s1_sample_xgb', 'pd')
    # data2 = load_data().get_test_s1('submission_s1_sample_mean_median', 'pd')
    #
    # data1['pred'] = (data1['pred'] + data2['pred']) / 2
    # data1.to_csv(load_data().get_project_path() + '/data/test_s1/submission_s1_sample_mean_median_xgb.csv', index=None,
    #              encoding='utf-8')
    #
    # print(time.clock() - start)
