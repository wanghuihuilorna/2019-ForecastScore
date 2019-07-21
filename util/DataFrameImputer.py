import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


# 数据转换类
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        self.fill = None

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],  # 计算中位数
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
