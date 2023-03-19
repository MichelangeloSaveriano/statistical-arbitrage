from preprocessing_base import PreprocessingBase


class NoPreprocessing(PreprocessingBase):
    def fit(self, train, test=None):
        return self

    def transform(self, test):
        return test

    def fit_transform(self, train):
        return train
