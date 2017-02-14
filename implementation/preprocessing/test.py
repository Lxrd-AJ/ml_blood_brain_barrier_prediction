
import sys, cPickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

trainset = sys.argv[1]
testset = sys.argv[2]
trainset = [mol for mol in Chem.SDMolSupplier(trainset) if mol is not None]
testset = [mol for mol in Chem.SDMolSupplier(testset) if mol is not None]

nms=[x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
print(len(nms))

trainDescrs = [calc.CalcDescriptors(x) for x in trainset]
testDescrs  = [calc.CalcDescriptors(x) for x in testset]
trainDescrs = np.array(trainDescrs)
testDescrs = np.array(testDescrs)

x_train_minmax = min_max_scaler.fit_transform( trainDescrs )
x_test_minmax = min_max_scaler.fit_transform( testDescrs )

classes={'(A) low':0,'(B) medium':1,'(C) high':1}
train_acts = np.array([classes[mol.GetProp("SOL_classification")] for mol in trainset],dtype="int")
test_acts = np.array([classes[mol.GetProp("SOL_classification")] for mol in testset],dtype="int")

dataset = ( (x_train_minmax, train_acts),(x_train_minmax, train_acts), (x_test_minmax, test_acts) )

f = open("rdk_sol_set_norm_descs.pkl", "wb")
cPickle.dump(dataset,f)
f.close()






"""
"""
import sys, cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn import metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

f = open(sys.argv[1], "rb")
train, valid, test = cPickle.load(f)

train_x, train_y = train
test_x, test_y = test

print "RANDPMFOREST"
nclf = RandomForestClassifier( n_estimators=100, max_depth=5, random_state=0, n_jobs=1 )
nclf = nclf.fit( train_x, train_y )
preds = nclf.predict( test_x )
print metrics.confusion_matrix(test_y, preds)
print metrics.classification_report(test_y, preds)
accuracy = nclf.score(test_x, test_y)
print accuracy

print "SVM"
clf_svm = svm.SVC( gamma=0.001, C=100. )
clf_svm = clf_svm.fit( train_x, train_y )
preds_SVM = clf_svm.predict( test_x )
print metrics.confusion_matrix( test_y, preds_SVM )
print metrics.classification_report( test_y, preds_SVM )
accuracy = clf_svm.score( test_x, test_y )

print accuracy

print "NB"
gnb = GaussianNB()
clf_NB = gnb.fit( train_x, train_y )
preds_NB = clf_NB.predict( test_x )
print metrics.confusion_matrix( test_y, preds_NB )
print metrics.classification_report( test_y, preds_NB )

#accuracy = preds_NB.score( test_x, test_y )
#print accuracy

print "RBM"
cls_svm2 = svm.SVC( gamma=0.001, C=100. )
rbm = BernoulliRBM(random_state = 0, verbose = True)
classifier = Pipeline( steps=[("rbm", rbm), ("cls_svm2", cls_svm2)] )
rbm.learning_rate = 0.06
rbm.n_iter = 20
rbm.n_compornents = 1000
classifier.fit(train_x, train_y)
pred_RBM = classifier.predict(test_x)
print metrics.confusion_matrix(test_y, pred_RBM)
print metrics.classification_report(test_y, pred_RBM)
accuracy = classifier.score( test_x, test_y )
print accuracy
