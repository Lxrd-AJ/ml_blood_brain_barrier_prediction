from models import custom_estimator, svm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import cross_val_score, train_test_split
from rdkit.Chem.AtomPairs import Pairs
import pandas as pd
import numpy as np
import os

LFClassifier = custom_estimator.LFClassifier

lf_clf = LFClassifier()
#lf_clf.sanity_check()

dataset_url = "./datasets/bbb_penetration_modified.txt"
desc_url = './datasets/bbb_penetration_molecular_descriptors.csv'

if not os.path.exists('./visualisations'):
    os.makedirs('./visualisations')


mol_df = pd.read_csv(desc_url)
x_cols = [col for col in mol_df.columns if col not in ['p_np','smiles']]

# Simple molecular descriptors
A = mol_df[x_cols]
y_a = mol_df.p_np

# Morgan Fingerprint dataset
molecules = [Chem.MolFromSmiles(smiles) for smiles in mol_df.smiles]
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(m,2) for m in molecules]
B = []
for fp in fingerprints:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp,arr)
    B.append(arr)
y_b = mol_df.p_np

print("Testing custom classifier")
# scores = cross_val_score(lf_clf, A, y_a,cv=5, scoring='accuracy')
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# X_train, X_test, y_train, y_test = train_test_split(A,y_a,random_state=10)
# lf_clf.fit(X_train, y_train)
# print("-> fit score: {:.2f}".format(lf_clf.score(X_test,y_test)))


svm_pipeline = svm.pipeline_svm(A,y_a,False)