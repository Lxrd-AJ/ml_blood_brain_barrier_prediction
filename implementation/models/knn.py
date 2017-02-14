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

def best_knn_pipeline(datasets):
    """
    Applies a feature union to each created pipeline for each dataset passed to it
    and returns the pipeline with the highest average score
    """
    molecular_descriptor_pipeline = knn_pipeline_molecular_descriptors(
        datasets["molecular_descriptors"][0],
        datasets["molecular_descriptors"][1]
    )

    fingerprint_pipeline = knn_pipeline_fingerprint(
        datasets["morgan_fingerprints"][0],
        datasets["morgan_fingerprints"][1]
    )

    pipelines = [molecular_descriptor_pipeline, fingerprint_pipeline]
    pipelines.sort(key=lambda ps: ps[0]) #sort by their cross val scores
    pipeline = pipelines[-1] # return the pipeline with the biggest score
    print("kNN classifier with the highest score => {}".format(pipeline))
    return pipeline[1] # return pipeline without cross val score

"""
- Compare different feature vectors {MoleculeDescriptors,Fingerprints}
- Compare different distance metrics
"""

def knn_pipeline_molecular_descriptors(X,y):
    """
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
    feature_union = FeatureUnion([
        ("scaler",MinMaxScaler(feature_range=(-1, 1))),
        #("poly_feat", PolynomialFeatures(degree=2))
        ])
    feature_union.fit(X,y)
    X = feature_union.transform(X)
    print("Training using the k-nearest neighbour classifier with feature union\n{}".format(feature_union))
    print("Using Simple Molecular Descriptors as the feature vectors")
    return knn_pipeline('knn_molecular_descriptors.png',X,y,("molecular_features",feature_union))

def knn_pipeline_fingerprint(X,y):
    """
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


def knn_pipeline(viz_name, X, y, feature_union, metric='minkowski',k_name=""):
    """
    TODO: Modify knn_pipeline to use cross_validation score instead
    Creates a Pipeline for 1 - 10 nearest neighbor and returns the pipeline with the
    highest test score
    """
    training_accuracy = []
    test_accuracy = []
    pipelines = []
    neighbor_range = range(1,11)
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
