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
from sklearn.pipeline import Pipeline,FeatureUnion
import matplotlib.pyplot as plt
import numpy as np

def pipeline_neural_network(X,y,isFingerprint=False):
    # split data into train+validation set and test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)
    # split train+validation set into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split( X_trainval, y_trainval, random_state=1)
    print("Size of training set: {} size of validation set: {} size of test set: {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

    pipeline = None
    feature_union = None
    
    if not isFingerprint:
        feature_union = FeatureUnion([
            ("min_max_scaler", MinMaxScaler(feature_range=(-1,1))),
            ("select_percentile", SelectPercentile(percentile=50))
        ])
    else:
        feature_union = FeatureUnion([
            ("select_percentile", SelectPercentile(percentile=50))
        ])
    
    pipeline = Pipeline([
        ('preprocessors', feature_union),
        ('neural_network', MLPClassifier())
    ])

    #Debugging purposes only 
    # for param,value in pipeline.get_params().items():
    #     print("Parameter = {:} \t Value= {:}".format(param,value))

    # Grid Searching to select the best parameters for the pipeline
    best_score = 0
    best_parameters = {}
    for hidden_layer in [(100,),(1000,)]:
        for alpha in [0.0001, 0.01, 1, 100]:
            for activation in ['identity', 'logistic', 'tanh', 'relu']:
                for solver in ['lbfgs', 'sgd', 'adam']:
                    for min_max in [(0,1), (-1,1), (-10,10)]:
                        for percentile in [10,50,100]:
                            parameters = {
                                'neural_network__hidden_layer_sizes': hidden_layer,
                                'neural_network__alpha': alpha,
                                'neural_network__activation': activation,
                                'neural_network__solver': solver
                            }
                            if not isFingerprint:
                                parameters['preprocessors__min_max_scaler__feature_range'] = min_max
                                parameters['preprocessors__select_percentile__percentile'] = percentile
                            
                            pipeline.set_params(**parameters)

                            for param, value in parameters.items():
                                print("-> Training MLP Classifier with {:} = {:}".format(param,value))
                            
                            pipeline.fit(X_train, y_train)
                            score = pipeline.score(X_valid, y_valid)

                            print("\t -> training score of {:.2f} \n".format(score))

                            if score > best_score:
                                best_score = score
                                best_parameters = parameters
        
        
    pipeline.set_params(**best_parameters)
    pipeline.fit(X_trainval, y_trainval)
    test_score = pipeline.score(X_test, y_test)
    print("Best score on validation set: {:.2f}".format(best_score))
    print("Best parameters: ", best_parameters)
    print("Test set score with best parameters: {:.2f}".format(test_score))

    return pipeline  




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




