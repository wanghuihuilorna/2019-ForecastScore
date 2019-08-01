import os
import numpy as np
import pandas as pd
import scipy.sparse as ss
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR

from util.tool import *
from util import tool
from util.load_data import load_data


def xgb_model(model_name, train_X, train_y, test_X, test_y=None):
    """
    xgb模型
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    """
    model_path = load_data().get_project_path() + '/model/' + model_name

    if os.path.exists(model_path):
        gbm = tool.load_model(model_path=model_path)
    else:
        gbm = xgb.XGBRegressor(max_depth=10, n_estimators=1000, learning_rate=0.01).fit(train_X, train_y)
        tool.save_model(gbm, model_path=model_path)

    if test_X is None:
        return None

    predictions = gbm.predict(test_X)

    if test_y is not None:
        test_y = np.array(test_y)
        predictions = np.array(predictions)
        predictions = np.floor(np.expm1(predictions))
        test_y = np.floor(np.expm1(test_y))
        rmse = np.sqrt(((test_y - predictions) ** 2).mean())

        return rmse, predictions
    else:
        predictions = np.floor(np.expm1(predictions))
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
    model_path = load_data().get_project_path() + '/model/' + model_name

    if os.path.exists(model_path):
        gbm = tool.load_model(model_path=model_path)
    else:
        gbm = lgb.LGBMRegressor(max_depth=10, n_estimators=1000, learning_rate=0.01).fit(train_X, train_y)
        tool.save_model(gbm, model_path=model_path)

    if test_X is None:
        return None

    predictions = gbm.predict(test_X)

    if test_y is not None:
        test_y = np.array(test_y)
        predictions = np.array(predictions)
        predictions = np.floor(np.expm1(predictions))
        test_y = np.floor(np.expm1(test_y))
        rmse = np.sqrt(((test_y - predictions) ** 2).mean())

        return rmse, predictions
    else:
        predictions = np.floor(np.expm1(predictions))
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
    model_path = load_data().get_project_path() + '/model/' + model_name

    if os.path.exists(model_path):
        svm_model = tool.load_model(model_path=model_path)
    else:
        svm_model = SVR(kernel='rbf', gamma='auto').fit(train_X, train_y)  # svm 输入的label必须是int类型。
        tool.save_model(svm_model, model_path=model_path)

    if test_X is None:
        return None

    predictions = svm_model.predict(test_X)

    if test_y is not None:
        test_y = np.array(test_y)
        predictions = np.array(predictions)
        predictions = np.floor(np.expm1(predictions))
        test_y = np.floor(np.expm1(test_y))
        rmse = np.sqrt(((test_y - predictions) ** 2).mean())

        return rmse, predictions
    else:
        predictions = np.floor(np.expm1(predictions))
        return predictions


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
    # index = 0
    # result = None
    # for course_id in ['course1', 'course2', 'course3', 'course4', 'course5', 'course6', 'course7', 'course8']:
    #     index += 1
    #     # 获取exam_score
    #     exam_score = get_exam_score('exam_score', course_id)
    #     # 获取exam_id
    #     exam_id = list(set(exam_score.exam_id))
    #     # 对exam_id进行排序
    #     exam_id.sort()
    #     # 获取训练集
    #     train_X = exam_score[exam_score.exam_id.isin(exam_id[:int(len(exam_id) - 2)])]
    #     # 获取测试集
    #     test_X = exam_score[exam_score.exam_id.isin(exam_id[int(len(exam_id) - 2):int(len(exam_id) - 1)])]
    #
    #     # 得分
    #     train_y = train_X.score
    #     # 得分
    #     test_y = test_X.score
    #
    #     del train_X['score']
    #     del test_X['score']
    #
    #     rmse, predictions = xgb_model('xgb_model' + str(index) + '.pkl', train_X, train_y, test_X, test_y)
    #
    #     print(rmse)
    #
    #     test_X.loc[:, 'predictions'] = predictions
    #     test_X.loc[:, 'course'] = course_id
    #
    #     tmp = test_X[['student_id', 'predictions', 'course']]
    #     tmp.reset_index(inplace=True, drop=True)
    #     if result is None:
    #         result = tmp
    #     else:
    #         result = pd.concat([result, tmp], axis=0)
    #
    # result.sort_values(by='student_id', inplace=True)
    # result.to_csv('predictions.csv', encoding='utf_8', index=None)

    #############################################################lgb测试#################################################################
    # # 获取course
    # course = get_course('course')
    # # 获取student
    # student = get_student('student')
    # # 获取all_knowledge
    # all_knowledge = get_all_knowledge('all_knowledge')
    #
    # index = 0
    # for course_id in ['course1', 'course2', 'course3', 'course4', 'course5', 'course6', 'course7', 'course8']:
    #     index += 1
    #     # 获取exam_score
    #     exam_score = get_exam_score('exam_score', course_id)
    #     # 获取exam_id
    #     exam_id = list(set(exam_score.exam_id))
    #     # 对exam_id进行排序
    #     exam_id.sort()
    #     # 获取训练集
    #     train_X = exam_score[exam_score.exam_id.isin(exam_id[:int(len(exam_id) - 2)])]
    #     # 获取测试集
    #     test_X = exam_score[exam_score.exam_id.isin(exam_id[int(len(exam_id) - 2):int(len(exam_id) - 1)])]
    #
    #     # 得分
    #     train_y = train_X.score
    #     # 得分
    #     test_y = test_X.score
    #
    #     del train_X['score']
    #     del test_X['score']
    #
    #     rmse, predictions = lgb_model('lgb_model' + str(index) + '.pkl', train_X, train_y, test_X, test_y)
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
    # reuslt = []
    # for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
    #     submission_s1 = get_submission_s1('submission_s1', 'course' + i)
    #
    #     exam_score = get_exam_score('exam_score', 'course' + i)
    #
    #     train_y = exam_score['score']
    #
    #     del exam_score['score']
    #
    #     columns = list(set(list(exam_score.columns) + list(submission_s1.columns)))
    #     columns.sort()
    #
    #     submission_s1 = submission_s1.reindex(columns=columns).fillna(0)
    #     exam_score = exam_score.reindex(columns=columns).fillna(0)
    #
    #     predictions = xgb_model(model_name='xgb_model' + i + '.pkl', train_X=exam_score, train_y=train_y,
    #                             test_X=submission_s1, test_y=None)
    #
    #     reuslt.extend(predictions.tolist())
    #
    # submit = load_data().get_test_s1('submission_s1', 'pd')
    # submit['pred'] = reuslt
    # submit.to_csv(load_data().get_project_path() + '/data/test_s1/submission_s1_sample_xgb.csv', index=None, encoding='utf-8')

    ####################################################lgb#################################################################
    # reuslt = []
    # for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
    #     submission_s1 = get_submission_s1('submission_s1', 'course' + i)
    #
    #     del submission_s1['pred']
    #
    #     exam_score = get_exam_score('exam_score', 'course' + i)
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
    #     submission_s1 = get_submission_s1('submission_s1', 'course' + i)
    #
    #     del submission_s1['pred']
    #
    #     exam_score = get_exam_score('exam_score', 'course' + i)
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
    #     train_X = get_exam_score('exam_score', 'course' + i)
    #     test_X = get_submission_s1('submission_s1', 'course' + i)
    #     del test_X['pred']
    #
    #     mean_value = tool.get_mean_value(train_X)
    #     median_value = tool.get_median_value(train_X)
    #     # mode_value = tool.get_mode_value(train_X)
    #     # max_value = tool.get_maximum_value(train_X)
    #     # min_value = tool.get_minimum_value(train_X)
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
    data1 = load_data().get_test_s1('submission_s1_sample_xgb', 'pd')
    data2 = load_data().get_test_s1('submission_s1_sample_baseline', 'pd')

    data1['pred'] = (data1['pred'] + data2['pred']) / 2
    data1.to_csv(load_data().get_project_path() + '/data/test_s1/submission_s1_sample_baseline_xgb.csv', index=None,
                 encoding='utf-8')

    print(time.clock() - start)
