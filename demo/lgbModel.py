import lightgbm as lgb
import pandas as pd
from sklearn import metrics
from scipy import sparse
from demo import preprocess
from util import load_data


class lgbodel:
    def __init__(self, feaName, cateFea=[], params={}):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 150,
            'max_depth': -1,
            'min_data_in_leaf': 350,
            # 'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 3,
            'verbose': 0,
            'seed': 0,
        }
        self.params.update(**params)
        self.feaName = feaName
        self.cateFea = cateFea

    def train(self, X, y, num_round=8000, valid_X=None, valid_y=None, early_stopping=10, verbose=True, params={}):
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        trainParam = self.params
        trainParam.update(params)
        if isinstance(valid_X, (pd.DataFrame, sparse.csr_matrix)):
            validData = trainData.create_valid(valid_X, label=valid_y)
            bst = lgb.train(params=trainParam, train_set=trainData, num_boost_round=num_round, valid_sets=[trainData, validData],
                            valid_names=['train', 'valid'], early_stopping_rounds=early_stopping, verbose_eval=verbose)
        else:
            bst = lgb.train(params=trainParam, train_set=trainData, valid_sets=trainData, num_boost_round=num_round,
                            verbose_eval=verbose)
        self.bst = bst
        return bst.best_iteration

    def cv(self, X, y, nfold=5, num_round=8000, early_stopping=10, verbose=True, params={}):
        trainParam = self.params
        trainParam.update(params)
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        result = lgb.cv(trainParam, trainData, feature_name=self.feaName, categorical_feature=self.cateFea,
                        num_boost_round=num_round, nfold=nfold, early_stopping_rounds=early_stopping,
                        verbose_eval=verbose)
        return result

    def predict(self, X):
        return self.bst.predict(X)

    def feaScore(self, show=True):
        scoreDf = pd.DataFrame({'fea': self.feaName, 'importance': self.bst.feature_importance()})
        scoreDf.sort_values(['importance'], ascending=False, inplace=True)
        if show:
            print(scoreDf[scoreDf.importance > 0])
        return scoreDf

    def gridSearch(self, X, y, valid_X, valid_y, nFold=5, verbose=0):
        paramsGrids = {
            'num_leaves': [20 * i for i in range(2, 10)],
            # 'max_depth': list(range(8,13)),
            # 'min_data_in_leaf': [50*i for i in range(2,10)],
            # 'bagging_fraction': [1-0.05*i for i in range(0,5)],
            # 'bagging_freq': list(range(0,10)),
        }

        def getEval(params):
            iter = self.train(X, y, valid_X=valid_X, valid_y=valid_y, params=params)
            print(self.predict(valid_X))
            return metrics.mean_squared_error(valid_y, self.predict(valid_X)), iter

        for k, v in paramsGrids.items():
            resultDf = pd.DataFrame({k: v})
            resultDf['metric_mean'] = list(map(lambda x: getEval({k: x}), v))
            print(resultDf)
        exit()


if __name__ == '__main__':
    # reuslt = []
    # for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
    #     submission_s1 = preprocess.get_submission_s1('submission_s1', [i])
    #
    #     del submission_s1['pred']
    #
    #     exam_score = preprocess.get_exam_score('exam_score', [i])
    #
    #     train_y = exam_score['score']
    #
    #     del exam_score['score']
    #
    #     predictions = preprocess.lgb_model(model_name='lgb_model_' + i + '.pkl', train_X=exam_score, train_y=train_y,
    #                                        test_X=submission_s1, test_y=None)
    #
    #     reuslt.extend(predictions.tolist())
    #
    # submit = load_data.load_data().get_test_s1('submission_s1', 'pd')
    # submit['pred'] = reuslt
    # submit.to_csv(load_data.load_data().get_project_path() + '/data/test_s1/submission_s1_sample_lgb.csv', index=None,
    #               encoding='utf-8')

    reuslt = []
    for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
        train_X = preprocess.get_exam_score('exam_score', [i])
        train_y = train_X['score']

        valid_X = train_X.iloc[int(train_X.shape[0]*0.9):]
        valid_y = train_y.iloc[int(train_y.shape[0]*0.9):]

        train_X = train_X[:int(train_X.shape[0]*0.9)]
        train_y = train_y[:int(train_y.shape[0]*0.9)]

        del train_X['score']
        del valid_X['score']
        fea = train_X.columns.to_list()

        # 训练模型
        model = lgbodel(fea)
        model.gridSearch(train_X, train_y.values, valid_X, valid_y.values)
        iterNum = model.train(train_X, train_y.values, valid_X=valid_X, valid_y=valid_y.values, params={'learning_rate': 0.01})

        # model.cv(dfX, dfy, nfold=5)
        # model.train(dfX, dfy, num_round=iterNum, params={'learning_rate': 0.02}, verbose=False)
        # model.feaScore()
        #
        # # 预测结果
        # predictDf = originDf[originDf.flag == -1][['instance_id', 'hour']]
        # predictDf['predicted_score'] = model.predict(testX)
        # print(predictDf[['instance_id', 'predicted_score']].describe())
        # print(predictDf[['instance_id', 'predicted_score']].head())
        # print(predictDf.groupby('hour')['predicted_score'].mean())
        # exportResult(predictDf[['instance_id', 'predicted_score']], "../result/lgb2.csv")

        # # 5折stacking
        # print('training oof...')
        # df2 = originDf[originDf.flag >= 0][['instance_id', 'hour', 'click']]
        # df2['predicted_score'], predictDf['predicted_score'] = getOof(model, dfX, dfy, testX)
        # print('cv5 valid loss:', metrics.log_loss(df2['click'], df2['predicted_score']))
        # print(predictDf[['instance_id', 'predicted_score']].describe())
        # print(predictDf[['instance_id', 'predicted_score']].head())
        # print(predictDf.groupby('hour')['predicted_score'].mean())
        # exportResult(df2[['instance_id', 'predicted_score']], "../result/lgb2_oof_train.csv")
        # exportResult(predictDf[['instance_id', 'predicted_score']], "../result/lgb2_oof_test.csv")
