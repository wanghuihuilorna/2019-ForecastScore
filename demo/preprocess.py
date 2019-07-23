import scipy.sparse as ss
import lightgbm as lgb
import numpy as np
import pandas as pd
from util.load_data import load_data
from util import util
from sklearn.svm import SVC
from sklearn import preprocessing


def get_course(filename, tag='pd'):
    """
    处理course.csv文件
    :param filename:
    :return:
    """
    df = load_data().get_train_s1(filename, tag)
    df = util.label_encoding(df, columns=[u'course_class'])

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


def lgb_model(model_name, train_X, train_y, test_X, test_y=None):
    """
    lgb模型
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    """
    gbm = lgb.LGBMRegressor(max_depth=5, n_estimators=500, learning_rate=0.01).fit(train_X, train_y)

    model_path = load_data().get_project_path() + '/model/' + model_name
    util.save_model(gbm, model_path=model_path)

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
    util.save_model(svm_model, model_path=model_path)

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


def get_exam_score(filename, course_id: list, tag='pd'):
    """
    处理exam_score.csv
    :param filename:
    :return:
    """
    exam_score = load_data().get_train_s1(filename, tag)

    student = get_student('student')
    course = get_course('course')

    # 合并性别
    exam_score = pd.merge(exam_score, student, how='left', on='student_id')
    # 合并course_class
    exam_score = pd.merge(exam_score, course, how='left', on='course')

    if len(course_id) != 0:
        course_list = ['course' + i for i in course_id]
        # 获取选中的course_id
        exam_score = exam_score[exam_score['course'].isin(course_list)]
        # 使用knowledge_point 替换 exam_id
        course_list = [i + '_exams' for i in course_list]

        for course_i in course_list:
            # 读取特定的course_exams.csv文件
            course_exams = get_course_exams(course_i)
            # 合并
            exam_score = pd.merge(exam_score, course_exams, how='left', on='exam_id')
            # 合并
            exam_score = merge_all_knowledge(exam_score, course_i)

        # 处理标签特征
        exam_score.index = list(exam_score['student_id'])
        del exam_score['student_id']
        exam_score = util.label_encoding(exam_score, ['course', 'exam_id'])

        return exam_score
    else:
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
        submission_s1.index = list(submission_s1['student_id'])
        del submission_s1['student_id']
        submission_s1 = util.label_encoding(submission_s1, ['course', 'exam_id'])

        return submission_s1
    else:
        return submission_s1


if __name__ == '__main__':
    # # 判断是否存在缺失值
    # test_s1_file_name = ['submission_s1']
    # train_s1_file_name = ['all_knowledge', 'course', 'course1_exams', 'course2_exams', 'course3_exams', 'course4_exams'
    #     , 'course5_exams', 'course6_exams', 'course7_exams', 'course8_exams', 'exam_score', 'student']
    # sample_s1_file_name = ['submission_s1_sample']
    #
    # ####################################################lgb测试#################################################################
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
    # del train_X['score']
    # del test_X['score']
    #
    # del train_X['exam_id']
    # del test_X['exam_id']
    #
    # # train_X = util.min_max_scaler(train_X)
    # # test_X = util.min_max_scaler(test_X)
    #
    # # print(train_X.shape)
    # # print([x for x in train_X.columns if (train_X[x].values != 0).any()])
    # # train_X = train_X([x for x in train_X.columns if (train_X[x].values != 0).any()])
    # # print(train_X.shape)
    # # print(test_X.shape)
    # # test_X = test_X([x for x in test_X.columns if (test_X[x].values != 0).any()])
    # # print(test_X.shape)
    #
    # rmse, predictions = lgb_model('lgb_model_1.pkl', train_X, train_y, test_X, test_y)
    # print(rmse)
    # print(predictions)
####################################################svm测试#################################################################
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
    # del train_X['score']
    # del test_X['score']
    #
    # del train_X['exam_id']
    # del test_X['exam_id']
    #
    # # train_X = util.min_max_scaler(train_X)
    # # test_X = util.min_max_scaler(test_X)
    #
    # rmse, predictions = svm_model('svm_model_1.pkl', train_X, train_y, test_X, test_y)
    # print(rmse)
    # print(predictions)
####################################################测试#################################################################


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
    reuslt = []
    for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
        print(i)
        submission_s1 = get_submission_s1('submission_s1', [i])

        del submission_s1['pred']

        exam_score = get_exam_score('exam_score', [i])

        train_y = exam_score['score']

        del exam_score['score']

        predictions = svm_model(model_name='svm_model_' + i + '.pkl', train_X=exam_score, train_y=train_y,
                                test_X=submission_s1, test_y=None)

        reuslt.extend(predictions.tolist())

    submit = load_data().get_test_s1('submission_s1', 'pd')
    submit['pred'] = reuslt
    submit.to_csv(load_data().get_project_path() + '/data/test_s1/submission_s1_sample_svm.csv', index=None, encoding='utf-8')
