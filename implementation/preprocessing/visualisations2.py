#from dataset import load_data
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler,label_binarize
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, KFold, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
import random  
import itertools
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 



def load_data(from_url):
    data = np.genfromtxt(from_url,dtype="i4,U256,U256,U256",
                         comments=None,skip_header=1,names=['num','name','p_np','smiles'],
                         converters={k: lambda x: x.decode("utf-8") for k in range(1,4,1)})
    fail_idx = []
    for idx,entry in enumerate(data):
        smiles = entry[3]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            fail_idx.append(idx)
            continue
    data = np.delete(data,fail_idx)
    print("Fail Count: ", len(fail_idx))
    print("{} molecules used in the calculations".format(len(data)))
    return data

data_url = "./../datasets/bbb_penetration_modified.txt"
descriptor_output_url = "./../datasets/bbb_penetration_molecular_descriptors.csv"

dataset = load_data(data_url)
mol_df = pd.read_csv(descriptor_output_url)
x_cols = [col for col in mol_df.columns if col not in ['p_np','smiles']]

# Simple molecular descriptors
A = mol_df[x_cols]
y_a = mol_df.p_np

random_entries = [dataset[x] for x in random.sample(range(0,2047), 6)]
random_molecules = [Chem.MolFromSmiles(entry[3]) for entry in random_entries]

"""
Draw a chart of random molecules and also create a file containing the chemical 
descriptors for the molecules 
=====
mol_images = Draw.MolsToGridImage(random_molecules, molsPerRow=3, subImgSize=(200,200),legends=[entry[1] for entry in random_entries])
mol_images.save("./../visualisations/random_molecules.png")
#create a file containing the chemical descriptors for the molecules
chem_descriptors = [desc[0] for desc in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(chem_descriptors) 
with open('./random_molecules_descriptors.csv','w') as f:
    f.write("Molecule_Name," + 
            ",".join(["{}".format(name) for name in calculator.descriptorNames]) + 
            "\n")
    for entry in random_entries:
        smiles = entry[3]
        molecule = Chem.MolFromSmiles(smiles)
        print("Calculating chemical descriptors for",smiles)
        f.write( entry[1] + "," +
                ", ".join(["{}".format(value)
                            for value in calculator.CalcDescriptors(molecule)]) +
                "\n")
    print("-> Finished writing to the file")
    print("Using",len(chem_descriptors), "chemical Descriptors")
"""

"""
Generating fingerprint information from the molecules
-> Generate images for each molecule with the smiles-format as its file name
-> Generate a txt file containing all the smiles and their fingerprint format  
===
print("Generating Fingerprint information")
random_entries = random_entries[:2]
random_molecules = random_molecules[:2]
# Generating images for each molecule 
with open('./random_fingerprints.txt','w') as f:
    for idx in range(0,2):
        filename = random_entries[idx][1] #The molecular name is the filename  
        Draw.MolToFile(random_molecules[idx],"./../visualisations/fingerprint_stub/" + filename + ".png")
        # Generate the entry in the txt file  
        f.write("{:} -> {:}\n".format(filename, random_entries[idx][3]))
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(random_molecules[idx],2)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint,arr)
        fingerprint = arr  
        f.write("{:}\n".format("=" * 10))
        f.write("{:}\n".format(list(fingerprint)))
"""
        
"""
Generate visualisations on how the neural network divides the data
===
print("\nGenerating visualisations for neural networks")
fig, axes = plt.subplots(2,4,figsize=(20,8))
pca = PCA(n_components=10, whiten=True)
scaler = MinMaxScaler(feature_range=(-1,1)) #StandardScaler()
for i, ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver="lbfgs", random_state=i, hidden_layer_sizes=[100,100])
    mlp.fit(A,y_a)

    plt.figure(figsize=(5, 5))
    plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
    plt.yticks(range(0,len(mol_df.columns),10), mol_df.columns)
    plt.xlabel("Columns in weight matrix")
    plt.ylabel("Input feature")
    plt.colorbar()
    plt.savefig("./../visualisations/neural_network_activation.png",format='png',dpi=700)
    plt.close()

    # X = scaler.fit_transform(A)
    # X_train = pca.fit_transform(X)
    # print(X_train[:,:2])
    # mlp = MLPClassifier(solver="lbfgs", random_state=i, hidden_layer_sizes=[100,100])
    # mlp.fit(X_train[:,:2],y_a)
    # mglearn.plots.plot_2d_separator(mlp,X_train[:,:2],fill=True,alpha=0.3, ax=ax)
    # mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_a, ax=ax)
    # #plt.scatter(X_train[:,0], X_train[:,1], y_a)
    # plt.savefig("./../visualisations/neural_network_classification.png",format='png',dpi=700)
"""  


brain_url = "./../brain_2017-05-01_bbb.pkl"
voting_clf = joblib.load(brain_url)



"""
ROC Curve plotting
===

y = label_binarize(y_a, classes=['p','n'])
X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=.5,random_state=0)
n_classes = y.shape[1]
y_score = voting_clf.fit(X_train, y_train).predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    tpr[i], fpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
#plt.show()
plt.savefig("./../visualisations/roc_curve.png",format='png',dpi=700)
"""



"""
Learning curve plot for the voting Classifier
===

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
train_sizes, train_scores, test_scores = learning_curve(voting_clf, A, y_a, cv=cv)
plt.figure()
plt.title("Learning curve Ensemble Classifier")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.ylim((0.7, 1.01))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.savefig("./../visualisations/learning_curve_ensemble_clf.png",format='png',dpi=700)
plt.show()
"""



"""
Box plot comparing all the classifiers in the pipeline and also the ensemble classifiers
===

models = []
results = []
names = []
models.extend(voting_clf.estimators)
models.append(("Ensemble", voting_clf))
for name, model in models:
	kfold = KFold(n_splits=10, random_state=9)
	cv_results = cross_val_score(model, A, y_a, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig("./../visualisations/model_comparison.png",format='png',dpi=700)
plt.show()
"""



"""
Confusion Matrix for the ensemble classifier 
===
"""
X_train, X_test, y_train, y_test = train_test_split(A, y_a, test_size=.3,random_state=10)
y_pred = voting_clf.fit(X_train, y_train).predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
classes = voting_clf.classes_
classes = ["Not Pass" if x == "n" else "Pass" for x in classes]
print(conf_matrix)
print(classes)
plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix of the Ensemble Classifier")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes) #, rotation=45
plt.yticks(tick_marks, classes)
thresh = conf_matrix.max() / 2.
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, conf_matrix[i, j],
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("./../visualisations/confusion_matrix.png",format='png',dpi=700)
plt.show()