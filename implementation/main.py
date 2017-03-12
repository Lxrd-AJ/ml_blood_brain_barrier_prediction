from preprocessing import dataset,visualisations,clustering
from models import knn,decision_tree,ensemble,svm,neural_network
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from rdkit.Chem.AtomPairs import Pairs
import os
import time
import pandas as pd
import numpy as np

#TODO: Use a Logging framework and log the important parameters to file

start_time = time.time()

dataset_url = "./datasets/bbb_penetration_modified.txt"
desc_url = './datasets/bbb_penetration_molecular_descriptors.csv'

if not os.path.exists('./visualisations'):
    os.makedirs('./visualisations')

# dataset.load_data(dataset_url, desc_url)

mol_df = pd.read_csv(desc_url)
x_cols = [col for col in mol_df.columns if col not in ['p_np','smiles']]

# Simple molecular descriptors
A = mol_df[x_cols]
y_a = mol_df.p_np

# Morgan Fingerprint dataset
molecules = [Chem.MolFromSmiles(smiles) for smiles in mol_df.smiles]
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in molecules]
B = []
print("Calculating Morgan Fingerprint descriptors ...")
for fp in fingerprints:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp,arr)
    B.append(arr)
y_b = mol_df.p_np

datasets = {
    "molecular_descriptors": (A,y_a),
    "morgan_fingerprints": (B,y_b)
}

"""
====================================================
visualisations
====================================================

visualisations.make_viz_per_feature_histogram(A,y_a)
visualisations.make_scatter_plot(A,y_a)
visualisations.make_heat_map_pca(A,y_a)
"""
# visualisations.make_scatter_plot_3D(A,y_a)
# visualisations.make_scatter_plot_3D(B,y_b)
# visualisations.make_scatter_plot(A,y_a)
# visualisations.make_scatter_plot(B,y_b)
# visualisations.make_heat_map_pca(A,y_a)


"""
====================================================
kNN Classifier
====================================================

print("Using the Simple Molecular descriptor dataset")
bagging_clf = ensemble.best_bagging_classifier(A,y_a)
print("Using the Morgan Fingerprint dataset")
bagging_clf = ensemble.best_bagging_classifier(B,y_b)
knn_pipeline = knn.best_knn_pipeline(datasets)
print("kNN classifier\n{}".format(knn_pipeline))
"""


"""
====================================================
Decision Tree Classifier
====================================================
print("Using Simple molecular Descriptors")
mol_tree = decision_tree.tree(A,y_a)
mol_extra_trees = decision_tree.extra_trees(A,y_a)
mol_forest = decision_tree.random_forest(A,y_a)
mol_adaboost = decision_tree.adaboost_trees(A,y_a)
mol_graidentboost = decision_tree.gradientboost_trees(A,y_a)
print("Using the Morgan fingerprint dataset")
fps_tree = decision_tree.tree(B,y_b)
fps_extra_trees = decision_tree.extra_trees(B,y_b)
fps_forest = decision_tree.random_forest(B,y_b)
fps_adaboost = decision_tree.adaboost_trees(B,y_b)
fps_gradientboost = decision_tree.gradientboost_trees(B,y_b)
"""



"""
====================================================
Support Vector Machines
====================================================

print("Using Simple molecular Descriptors")
viz_title = "./visualisations/mol_decision_svm.png"
mol_svm = svm.classifier_svc(A,y_a,should_scale=True,viz_title=viz_title)
mol_svm = svm.classifier_svc(A,y_a,should_scale=False,viz_title=viz_title)

print("Using the Morgan fingerprint dataset")
viz_title = "./visualisations/fps_decision_svm.png"
fps_svm = svm.classifier_svc(B,y_b,should_scale=False,viz_title=viz_title)
"""



"""
====================================================
Neural Networks
====================================================
"""
print("Using Simple molecular Descriptors")
mol_mlp = neural_network.mlp_classifier(A,y_a)
print("*****" * 5, "Attempting Automatic Feature Selection", "*****"*5)
print("Using Simple molecular Descriptors")
viz_name = "neural_network_smd_univariate_matrix.png"
mol_mlp = neural_network.mlp_classifier_univariate(A,y_a,viz_name,should_scale=True)

viz_name = "neural_network_fps_univariate_matrix.png"
print("Using the Morgan fingerprint dataset")
# fps_mlp = neural_network.mlp_classifier(B,y_b)
fps_mlp = neural_network.mlp_classifier_univariate(B,y_b,viz_name)



"""
====================================================
Clustering
====================================================
clustering.make_cluster_plot(A,y_a,"visualisations/kmeans_smd_scatter.png")
print("Cluster Morgan Fingerprint data")
clustering.make_cluster_plot(B,y_b,"visualisations/kmeans_morgan_scatter.png")

print("Using Simple molecular Descriptors")
clustering.make_agglomerative_cluster_plot(A,y_a,should_scale=True,
                                           viz_name="visualisations/agg_smd_scatter.png")
print("Using the Morgan fingerprint dataset")
clustering.make_agglomerative_cluster_plot(B,y_b,should_scale=False,
                                           viz_name="visualisations/agg_morgan_scatter.png")
print("Simple Molecular Descriptors")
clustering.dbscan(A,y_a,"",True)
print("Morgan Fingerprint")
clustering.dbscan(B,y_b,"",False)
"""

"""
Create an Ensemble voting classifier. See http://scikit-learn.org/stable/modules/ensemble.html#majority-class-labels-majority-hard-voting for more information
""" 
# smd_ensemble_classifier = VotingClassifier(
#     estimators=[
#         ("smd_neural_net", mol_mlp),
#         ("smd_support_vector", mol_svm)
#     ], voting='hard'
# )
# # Testing the ensemble classifier
# # score = cross_val_score(smd_ensemble_classifier, A, y_a, cv=5, scoring='accuracy')
# # print("\n\n-> Cross Validation score of Ensemble Classifier = {:.1f}% (+/- {:.2f}%)".format(score.mean() * 100, score.std()))
# print("\n\n Ensemble Voting Classifier\n")
# for clf, label in zip([mol_mlp,mol_svm, smd_ensemble_classifier], ['Neural Net', 'Support Vector Machine', 'Ensemble']):
#     scores = cross_val_score(clf, A, y_a, cv=5, scoring='accuracy')
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

print("Program execution took {} (s)".format(time.time() - start_time))



"""
conda install -c rdkit rdkit
conda install scikit-learn
pip install graphviz
"""
