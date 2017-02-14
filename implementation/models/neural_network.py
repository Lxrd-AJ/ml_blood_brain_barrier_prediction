# @Author: AJ Ibraheem <AJ>
# @Date:   2016-11-12T14:11:52+00:00
# @Email:  ibraheemaj@icloud.com
# @Last modified by:   AJ
# @Last modified time: 2016-11-12T14:11:57+00:00

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile
import matplotlib.pyplot as plt
import numpy as np

def mlp_classifier(X,y):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    #TODO: Decide whether scale X = scaler.fit_transform(X)

    clf = MLPClassifier()
    cross_shuffle_split = StratifiedShuffleSplit(test_size=0.5,train_size=0.5,n_splits=10)
    score = cross_val_score(clf,X,y,cv=cross_shuffle_split).mean()
    print("-> Training accuracy of the Neural Network (MLP) classifier = {:.1f}%"
          .format(score * 100))
    return clf

def mlp_classifier_univariate(X,y,viz_name,should_scale=False):
    """
    Automatic feature selection using the analysis of variance technique
    """
    if should_scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    select = SelectPercentile(percentile=50)
    select.fit(X_train, y_train)
    mask = select.get_support()
    plt.matshow(mask.reshape(1,-1), cmap='gray_r')
    plt.xlabel("Index of features")
    plt.savefig("./visualisations/" + viz_name,format='png',dpi=500)
    plt.close()
    X_train_ = select.transform(X_train)
    X_test_ = select.transform(X_test)
    clf = MLPClassifier()
    clf.fit(X_train_, y_train)
    print("-> Testing accuracy of the Neural Network using univariate analysis = {:.1f}%".format(clf.score(X_test_, y_test) * 100))

    X_mod = np.vstack((X_train_,X_test_))
    y_mod = np.hstack((y_train, y_test))
    score = cross_val_score(clf,X_mod,y_mod,cv=5).mean()
    print("-> Cross Validation score = {:.1f}%".format(score * 100))
    return clf
