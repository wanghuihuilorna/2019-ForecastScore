import numpy as np  # linear algebra
from datetime import datetime
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from util.tool import get_exam_score, get_submission_s1
from util.load_data import load_data


def get_model(X, y, X_sub):
    # 设置k折交叉验证的参数。
    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

    # 定义均方根对数误差（Root Mean Squared Logarithmic Error ，RMSLE）
    def rmsle(y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

    # 创建模型评分函数，根据不同模型的表现打分
    # cv表示Cross-validation,交叉验证的意思。
    def cv_rmse(model, X=X):
        rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
        return (rmse)

    # 个体机器学习模型的创建（即模型声明和参数设置）-【开始】
    alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
    alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
    e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
    e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

    # 定义ridge岭回归模型（使用二范数作为正则化项。不论是使用一范数还是二范数，正则化项的引入均是为了降低过拟合风险。）
    # 注：正则化项如果使用二范数，那么对于任何需要寻优的参数值，在寻优终止时，它都无法将某些参数值变为严格的0，尽管某些参数估计值变得非常小以至于可以忽略。即使用二范数会保留变量的所有信息，不会进行类似PCA的变量凸显。
    # 注：正则化项如果使用一范数，它比L2范数更易于获得“稀疏(sparse)”解，即它的求解结果会有更多的零分量。
    ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

    # 定义LASSO收缩模型（使用L1范数作为正则化项）（由于对目标函数的求解结果中将得到很多的零分量，它也被称为收缩模型。）
    # 注：正则化项如果使用二范数，那么对于任何需要寻优的参数值，在寻优终止时，它都无法将某些参数值变为严格的0，尽管某些参数估计值变得非常小以至于可以忽略。即使用二范数会保留变量的所有信息，不会进行类似PCA的变量凸显。
    # 注：正则化项如果使用一范数，它比L2范数更易于获得“稀疏(sparse)”解，即它的求解结果会有更多的零分量。
    lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

    # 定义elastic net弹性网络模型（弹性网络实际上是结合了岭回归和lasso的特点，同时使用了L1和L2作为正则化项。）
    elasticnet = make_pipeline(RobustScaler(),
                               ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))

    # 定义SVM支持向量机模型
    svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))

    # 定义GB梯度提升模型（展开到一阶导数）
    gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)

    # 定义lightgbm模型
    lightgbm = LGBMRegressor(objective='regression',
                             num_leaves=4,
                             learning_rate=0.01,
                             n_estimators=5000,
                             max_bin=200,
                             bagging_fraction=0.75,
                             bagging_freq=5,
                             bagging_seed=7,
                             feature_fraction=0.2,
                             feature_fraction_seed=7,
                             verbose=-1,
                             # min_data_in_leaf=2,
                             # min_sum_hessian_in_leaf=11
                             )

    # 定义xgboost模型（展开到二阶导数）
    xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                           max_depth=3, min_child_weight=0,
                           gamma=0, subsample=0.7,
                           colsample_bytree=0.7,
                           objective='reg:linear', nthread=-1,
                           scale_pos_weight=1, seed=27,
                           reg_alpha=0.00006,
                           param={'objective': 'reg:squarederror',  # Specify multiclass classification
                                  'tree_method': 'gpu_hist'  # Use GPU accelerated algorithm
                                  })

    # 个体机器学习模型的创建（即模型声明和参数设置）-【结束】

    # 集成多个个体学习器-【开始】
    stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                    meta_regressor=xgboost,
                                    use_features_in_secondary=True)  # regressors=(...)中并没有纳入前面的svr模型
    # 集成多个个体学习器-【结束】

    # 进行交叉验证打分-【开始】
    # 进行交叉验证，并对不同模型的表现打分
    # （由于是交叉验证，将使用不同的数据集对同一模型进行评分，故每个模型对应一个得分序列。展示模型得分序列的平均分、标准差）
    print('进行交叉验证，计算不同模型的得分TEST score on CV')

    # 打印二范数rideg岭回归模型的得分
    score = cv_rmse(ridge)
    print("二范数rideg岭回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    # 打印一范数LASSO收缩模型的得分
    score = cv_rmse(lasso)
    print("一范数LASSO收缩模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    # 打印elastic net弹性网络模型的得分
    score = cv_rmse(elasticnet)
    print("elastic net弹性网络模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    # 打印SVR支持向量机模型的得分
    score = cv_rmse(svr)
    print("SVR支持向量机模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    # 打印lightgbm轻梯度提升模型的得分
    score = cv_rmse(lightgbm)
    print("lightgbm轻梯度提升模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    # 打印gbr梯度提升回归模型的得分
    score = cv_rmse(gbr)
    print("gbr梯度提升回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

    # 打印xgboost模型的得分
    score = cv_rmse(xgboost)
    print("xgboost模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
    # 进行交叉验证打分-【结束】

    # 使用训练数据特征矩阵作为输入，训练数据对数处理后的预测房价作为输出，进行各个模型的训练-【开始】
    # 开始集合所有模型，使用stacking方法
    print('进行模型参数训练 START Fit')

    print(datetime.now(), '对stack_gen集成器模型进行参数训练')
    stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

    print(datetime.now(), '对elasticnet弹性网络模型进行参数训练')
    elastic_model_full_data = elasticnet.fit(X, y)

    print(datetime.now(), '对一范数lasso收缩模型进行参数训练')
    lasso_model_full_data = lasso.fit(X, y)

    print(datetime.now(), '对二范数ridge岭回归模型进行参数训练')
    ridge_model_full_data = ridge.fit(X, y)

    print(datetime.now(), '对svr支持向量机模型进行参数训练')
    svr_model_full_data = svr.fit(X, y)

    print(datetime.now(), '对GradientBoosting梯度提升模型进行参数训练')
    gbr_model_full_data = gbr.fit(X, y)

    print(datetime.now(), '对xgboost二阶梯度提升模型进行参数训练')
    xgb_model_full_data = xgboost.fit(X, y)

    print(datetime.now(), '对lightgbm轻梯度提升模型进行参数训练')
    lgb_model_full_data = lightgbm.fit(X, y)

    # 使用训练数据特征矩阵作为输入，训练数据对数处理后的预测房价作为输出，进行各个模型的训练-【结束】

    # 进行交叉验证打分-【结束】

    # 定义个体学习器的预测值融合函数，检测预测值融合策略的效果-【开始】
    # 综合多个模型产生的预测值，作为多模型组合学习器的预测值
    def blend_models_predict(X):
        return ((0.05 * elastic_model_full_data.predict(X)) +
                (0.05 * lasso_model_full_data.predict(X)) +
                (0.05 * ridge_model_full_data.predict(X)) +
                (0.05 * svr_model_full_data.predict(X)) +
                (0.05 * gbr_model_full_data.predict(X)) +
                (0.3 * xgb_model_full_data.predict(X)) +
                (0.15 * lgb_model_full_data.predict(X)) +
                (0.3 * stack_gen_model.predict(np.array(X))))

    # 打印在上述模型配比下，多模型组合学习器的均方根对数误差（Root Mean Squared Logarithmic Error ，RMSLE）
    # 使用训练数据对创造的模型进行k折交叉验证，以训练创造出的模型的参数配置。交叉验证训练过程结束后，将得到模型的参数配置。使用得出的参数配置下，在全体训练数据上进行验证，验证模型对全体训练数据重构的误差。
    print('融合后的训练模型对原数据重构时的均方根对数误差RMSLE score on train data:')
    print(rmsle(y, blend_models_predict(X)))
    # 定义个体学习器的预测值融合函数，检测预测值融合策略的效果-【结束】

    # 将测试集的特征矩阵作为输入，传入训练好的模型，得出的输出写入.csv文件的第2列-【开始】
    # print('使用测试集特征进行房价预测 Predict submission', datetime.now(), )
    # submission = pd.read_csv("../data/sample_submission.csv")
    # # 函数注释：.iloc[:,1]是基于索引位来选取数据集，[索引1:索引2]，左闭右开。
    # submission.iloc[:, 1] = np.floor(np.expm1(blend_models_predict(X_sub)))
    # 将测试集的特征矩阵作为输入，传入训练好的模型，得出的输出写入.csv文件的第2列-【结束】

    return np.floor(np.expm1(blend_models_predict(X_sub)))


def main():
    reuslt = []
    for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
        # 训练集
        train_X = get_exam_score('exam_score', 'course' + i)
        train_y = train_X['score']
        del train_X['score']

        # 测试集
        test_X = get_submission_s1('submission_s1', 'course' + i)

        # 规范columns
        columns = list(set(list(train_X.columns) + list(test_X.columns)))
        columns.sort()
        test_X = test_X.reindex(columns=columns).fillna(0)
        train_X = train_X.reindex(columns=columns).fillna(0)

        # 保存预测结果
        reuslt.extend(get_model(train_X, train_y, test_X).tolist())

    submit = load_data().get_test_s1('submission_s1', 'pd')
    submit['pred'] = reuslt
    submit.to_csv(load_data().get_project_path() + '/data/test_s1/submission_s1_sample_stack.csv', index=None,
                  encoding='utf-8')


if __name__ == '__main__':
    main()
