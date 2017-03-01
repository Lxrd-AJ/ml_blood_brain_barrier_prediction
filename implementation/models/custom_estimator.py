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
        self.is_fingerprint = is_fingerprint
        self.classifiers_ = {
            "support_vector_machine": {
                "classifier": SVC(kernel='linear', probability=True, class_weight="balanced"),
                "pca": PCA(),
                "scaler": StandardScaler()
            }
        }

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        #Fit each classifier
        for clf in self.classifiers_.keys() :
            if clf == "support_vector_machine":
                self.classifiers_[clf] = self.__fit_support_vector_machine()
        return self
        

    """
    - TODO: Find a way to compare the results from all the classifiers
        predict_proba returns [[ 0.18299209  0.81700791]
                [ 0.17936999  0.82063001]
                [ 0.06809605  0.93190395]]
        which determines the confidence of class [n,p].
        OR use a Voting classifier underneath as described in http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html 
        OR Use a well developed pipeline(http://scikit-learn.org/stable/modules/pipeline.html) in combination with the voting classifier
    """ 
    def predict(self,X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        results = [] #An array of tuples (prediction,score)
        for clf in self.classifiers_.keys():
            if clf == "support_vector_machine":
                clf_y = self.__predict_support_vector_machine(X)
                print("Prediction result {:}".format(clf_y))
                results.append(clf_y)
        return results[0] #TODO: Remove hack later


    def __predict_support_vector_machine(self, X):
        pca = self.classifiers_["support_vector_machine"]["pca"]
        scaler = self.classifiers_["support_vector_machine"]["scaler"]
        classifier = self.classifiers_["support_vector_machine"]["classifier"]
        if not self.is_fingerprint:
            X = scaler.transform(X)
            X = pca.transform(X)
        print(classifier.predict_proba(X))
        return classifier.predict(X)

    def __fit_support_vector_machine(self):
        # pca = PCA(n_components='mle',svd_solver='full')
        pca = PCA(n_components=50)
        scaler = StandardScaler()
        X = self.X_
        if not self.is_fingerprint:
            X = scaler.fit_transform(X)
            X = pca.fit_transform(X)
        clf = SVC(kernel='poly', probability=True, class_weight="balanced")
        clf.fit(X, self.y_)
        # print("-> Trained Support vector classifier with train fit score: {:.2f}".format(clf.score(X,self.y_)))
        # print("--> Fitted classifier {:}".format(clf))
        return {
            "classifier": clf,
            "pca": pca,
            "scaler": scaler
        }


    def sanity_check(self):
        return check_estimator(LFClassifier)