"""
Custom Estimator according to http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
All the previously trained models would be combined here and the custom estimator here would be used for predictions
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC,LinearSVC
from sklearn.utils.estimator_checks import check_estimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class LFClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, is_fingerprint=False):
        self.X_ = []
        self.y_ = []
        self.classes_ = []
        self.is_fingerprint_ = is_fingerprint;
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
                self.classifiers_[clf] = self.__fit_support_vector_machine(!self.is_fingerprint)
        

    """
    - TODO: Find a way to compare the results from all the classifiers
    """ 
    def predict(self,X):
        results = [] #An array of tuples (prediction,score)
        for clf in self.classifiers_.keys():
            if clf == "support_vector_machine":
                clf_y = self.classifiers_[clf].predict(X)
                results.append(clf_y)
        return results[0] #TODO: Remove hack later


    def __fit_support_vector_machine(self,should_scale=True):
        pca = PCA(n_components=50)
        scaler = StandardScaler()
        if should_scale:
            X = scaler.fit_transform(self.X_)
        X = pca.fit_transform(X)
        clf = SVC(kernel='linear', probability=True, class_weight="balanced")
        clf.fit(X, self.y_)
        print("-> Trained Support vector classifier with fit score: {:.2f}".format(clf.score(X,self.y_)))
        return clf


    def sanity_check():
        return check_estimator(LFClassifier)