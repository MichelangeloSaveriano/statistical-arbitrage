from .preprocessing_base import PreprocessingBase
from sklearn.linear_model import ElasticNet


class ResidualsPreprocessing(PreprocessingBase):
    def __init__(self, factors, alpha_elastic_net=3e-3):
        self._model = None
        self._factors = factors.copy()
        self._factors.columns += '_factor'
        self._alpha_elastic_net = alpha_elastic_net

    def fit(self, train, test=None):
        train_factors_merged = train.merge(self._factors, right_index=True, left_index=True)
        factors_data = train_factors_merged[self._factors.columns]
        train_data = train_factors_merged.drop(columns=self._factors.columns)

        self._model = ElasticNet(alpha=self._alpha_elastic_net, fit_intercept=False).fit(factors_data, train_data)

        return self

    def transform(self, test):
        if self._model is None:
            raise Exception('Model not yet trained!')

        test_factors_merged = test.merge(self._factors, right_index=True, left_index=True)
        factors_data = test_factors_merged[self._factors.columns]
        test_data = test_factors_merged.drop(columns=self._factors.columns)

        test_residuals = test_data - self._model.predict(factors_data)

        return test_residuals

    def fit_transform(self, train):
        self.fit(train)
        return self.transform(train)
