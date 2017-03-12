from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.KNN.DistFunctions import TanimotoDist
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pipeline_knn(X,y,isFingerprint=False):
    # split data into train+validation set and test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)
    # split train+validation set into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split( X_trainval, y_trainval, random_state=1)
    print("Size of training set: {} size of validation set: {} size of test set: {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

    pipeline = None
    feature_union = FeatureUnion([
            ("scaler",MinMaxScaler(feature_range=(-1, 1))),
            #("poly_feat", PolynomialFeatures(degree=2))
        ])
    
    if not isFingerprint:
        pipeline = Pipeline([
            ("feature_union",feature_union),
            ("knn_classifier",KNeighborsClassifier(metric='minkowski',leaf_size=40,n_jobs=-1))
        ])
    else:
        pipeline = Pipeline([
            ("knn_classifier",KNeighborsClassifier(metric='dice',leaf_size=40,n_jobs=-1))
        ])

    # Grid Searching to select the best parameters for the pipeline
    best_score = 0
    best_parameters = {}
    for n_neighbors in [5,10,15]:
        for weights in ['uniform','distance']:
            for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
                for leaf_size in [30,50,70]:
                    parameters = {
                        'knn_classifier__n_neighbors': n_neighbors,
                        'knn_classifier__weights': weights,
                        'knn_classifier__algorithm': algorithm,
                        'knn_classifier__leaf_size': leaf_size
                    }
    
                    pipeline.set_params(**parameters)

                    for param, value in parameters.items():
                        print("-> Training kNN Classifier with {:} = {:}".format(param,value))
                    
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




def knn_classifier(viz_name, X, y, metric='minkowski',k_name=""):
    """
    TODO: Refactor this function to work indepe
    TODO: Modify knn_pipeline to use cross_validation score instead
    Creates a Pipeline for 1 - 10 nearest neighbor and returns the pipeline with the
    highest test score
    """
    training_accuracy = []
    test_accuracy = []
    pipelines = []
    neighbor_range = range(1,11)
    feature_union = FeatureUnion([
            ("scaler",MinMaxScaler(feature_range=(-1, 1))),
            #("poly_feat", PolynomialFeatures(degree=2))
        ])
    for i in neighbor_range:
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=i)
        print("Creating Pipeline using {} neighbor(s)".format(i))
        if feature_union is not None:
            pipeline = Pipeline([
                feature_union, #feature_union should be a tuple
                ("knn_classifier",KNeighborsClassifier(n_neighbors=i, metric=metric,n_jobs=-1,leaf_size=40))
                ])
        else:
            pipeline = Pipeline([
                ("knn_classifier",KNeighborsClassifier(n_neighbors=i, metric=metric,n_jobs=-1,leaf_size=40))
                ])
        pipeline.fit(X_train,y_train)
        trn_accuracy = pipeline.score(X_train,y_train)
        tst_accuracy = pipeline.score(X_test,y_test)
        avg_score = cross_val_score(pipeline, X,y).mean()
        print("-> Training accuracy of the knn classifier: {:.2f}".format(trn_accuracy))
        print("-> Test accuracy of the knn classifier: {:.2f}".format(tst_accuracy))
        print("-> Cross Validation Score of the kNN Classifier: {:.2f}".format(avg_score))
        training_accuracy.append(trn_accuracy)
        test_accuracy.append(tst_accuracy)
        pipelines.append((avg_score,pipeline))
    plt.plot(neighbor_range, training_accuracy, label="Training accuracy")
    plt.plot(neighbor_range, test_accuracy, label="Test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.savefig("./visualisations/" + viz_name,format='png',dpi=500)
    plt.close()

    pipelines.sort(key=lambda ts: ts[0]) #take the small training scores
    return pipelines[0]

def knn_pipeline_molecular_descriptors(X,y):
    """
    TODO: Deprecate function
    Using K-Nearest neighbour
    Predicts the probability of a molecule passing throught the blood brain barrier based on
    molecules similar to the target molecule.
    Dataset is rescaled to achieve the best result for simple molecular descriptors
    Returns the
    - The feature vectors used here are the simple molecular descriptors
    - notes:
        * Without preprocessing the knn classifier performs horribly
            * The best preprocessor so far is the MinMaxScaler
        * Optimal prediction value at n = 9 or 10
    - todo:
        * Fit the feature union only on the X_train data and not X
    """
    
    feature_union.fit(X,y)
    X = feature_union.transform(X)
    print("Training using the k-nearest neighbour classifier with feature union\n{}".format(feature_union))
    print("Using Simple Molecular Descriptors as the feature vectors")
    return knn_pipeline('knn_molecular_descriptors.png',X,y,("molecular_features",feature_union))

def knn_pipeline_fingerprint(X,y):
    """
    TODO: Deprecate function
    k-nearest neighbour classifier using the molecular fingerprint as the feature vector.
        - Fingerprints are calculated by using a kernel function to extract features from a molecule
            which are then hashed to create a bit vector.
        - More information online at
            http://rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf
        - [x] The Tanimoto similarity / Dice measure is used as a distance metric for the knn classifier as they provide better
            distance metrics than minkowski
    """
    print("\n","===" * 15)
    print("Testing using the k-nearest neighbour classifier")
    print("Using the Morgan Fingerprints as the feature vectors")
    return knn_pipeline('knn_morgan_dice_fingerprint.png',X,y,feature_union=None,metric='dice')


def tanimoto_dist(x1,y1):
    """
    A metric function that calculates the distance between molecules using their
    tanimoto or Jaccard similarity https://en.wikipedia.org/wiki/Chemical_similarity
    - todo: Use this instead http://www.rdkit.org/Python_Docs/rdkit.ML.KNN.DistFunctions-module.html#TanimotoDist
    # fp_str = fp.ToBitString()

    x1_str = ''.join(str(int(x)) for x in x1)
    y1_str = ''.join(str(int(x)) for x in y1)
    fp1 = DataStructs.CreateFromBitString(x1_str)
    fp2 = DataStructs.CreateFromBitString(y1_str)
    dist = 1 - DataStructs.TanimotoSimilarity(fp1,fp2)
    print("Tanimoto distance for x1 and y1 = {}".format(dist))
    return dist
    """
    return TanimotoDist(x1,y1,range(len(x1)))
