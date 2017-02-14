"""
Custom Estimator according to http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
All the previously trained models would be combined here and the custom estimator here would be used for predictions
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC,LinearSVC

class LFClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, is_fingerprint=False):
        self.X_ = []
        self.y_ = []
        self.classes_ = []
        self.classifiers_ = {
            "support_vector_machine": SVC(kernel='linear',probability=True,class_weight='balanced')
        }

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        #Fit each classifier
        for clf in self.classifiers_.keys() :
            if clf == "support_vector_machine":


    def __fit_support_vector_machine(self,should_scale=True):
        print()