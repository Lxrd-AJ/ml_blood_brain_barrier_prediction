# @Author: AJ Ibraheem <AJ>
# @Date:   2016-11-11T21:51:59+00:00
# @Email:  ibraheemaj@icloud.com
# @Last modified by:   AJ
# @Last modified time: 2016-11-11T21:52:04+00:00
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random

def best_bagging_classifier(X,y):
    #TODO: Move to KNN
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=random.randint(1,50))
    bagging_clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    bagging_clf.fit(X_train,y_train)
    print("===" * 15)
    print("-> Bagging classifier for k-nearest neighbors {}".format(bagging_clf.score(X_test,y_test)))

    return bagging_clf
