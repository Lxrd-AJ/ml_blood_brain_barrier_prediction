from knn import best_knn_pipeline
from svm import pipeline_svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import cross_val_score, train_test_split
from rdkit.Chem.AtomPairs import Pairs
import pandas as pd
import numpy as np
import os

dataset_url = "./../datasets/bbb_penetration_modified.txt"
desc_url = './../datasets/bbb_penetration_molecular_descriptors.csv'

if not os.path.exists('./../visualisations'):
    os.makedirs('./../visualisations')


mol_df = pd.read_csv(desc_url)
x_cols = [col for col in mol_df.columns if col not in ['p_np','smiles']]

# Simple molecular descriptors
A = mol_df[x_cols]
y_a = mol_df.p_np

"""
Collect all the pipelines and combine into the voting classifier
"""
svm_pipeline = pipeline_svm(A,y_a)

#Soft voting uses the class label based on the argmax of the sums of the predicted probabilities, which is recommended for an ensemble of well-calibrated classifiers.
# TODO: Explore if its possible to make a voting classifier that consists of 2 voting classifiers, 1 for fingerprint pipelines and the other for simple molecular descriptor pipelines
print("\n\nTraining the voting classifier")
voting_clf = VotingClassifier(estimators=[
    ('svm', svm_pipeline)
], voting="soft", n_jobs=-1)

scores = cross_val_score(voting_clf, A, y_a,cv=5, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

