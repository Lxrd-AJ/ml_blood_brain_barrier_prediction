# @Author: AJ Ibraheem <AJ>
# @Date:   2016-11-12T12:23:30+00:00
# @Email:  ibraheemaj@icloud.com
# @Last modified by:   AJ
# @Last modified time: 2016-11-12T12:23:35+00:00

from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

def classifier_svc(X,y,should_scale=False,viz_title="./visualisations/decision_svm.png"):
    pca = PCA(n_components=50)
    scaler = StandardScaler() #MinMaxScaler(feature_range=(-1, 1))
    classifiers = []

    if should_scale:
        X = scaler.fit_transform(X)
    X = pca.fit_transform(X)

    for kernel in ['linear','rbf','poly','sigmoid']:
        clf = SVC(kernel=kernel,probability=True,class_weight='balanced')
        score = cross_val_score(clf,X,y).mean()
        print("-> Training accuracy of the support vector machines using {} kernel = {:.1f}%"
              .format(kernel, score * 100))
        classifiers.append(clf)

    clf = LinearSVC(class_weight='balanced')
    score = cross_val_score(clf,X,y).mean()
    print("-> Training accuracy of the support vector machines using a LinearSVC = {:.1f}%"
          .format(score * 100))
    classifiers.append(clf)

    titles = ['SVC with linear kernel','SVC with RBF Kernel',
              'SVC with polynomial (degree 3) kernel','SVC with sigmoid kernel',
              'LinearSVC (Linear kernel)']

    #Use only the first two Principal components
    #TODO: Move plot code elsewhere
    # plot_decision_surface(X[:,:2],y,classifiers,titles,viz_title)
    return SVC(kernel='linear',probability=True,class_weight='balanced') #TODO: Decide later which classifier to return

def plot_decision_surface(X,y,classifiers,titles,viz_name):
    y = [1 if x == 'p' else 0 for x in y]
    for clf in classifiers:
        clf.fit(X,y)
    step_size = 0.5
    # Plot mesh
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,step_size),
                         np.arange(y_min,y_max,step_size))

    plt.close()
    for idx, clf in enumerate(classifiers):
        print("Plotting decision boundary for {}".format(clf))
        plt.subplot(3,2, idx+1)
        plt.subplots_adjust(wspace=1,hspace=1)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=0.8)
        plt.scatter(X[:,0], X[:,1],c=y,cmap=plt.cm.coolwarm)
        plt.xlabel("First Component")
        plt.ylabel("Second Component")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[idx])

    plt.savefig(viz_name,format='png',dpi=1080)
    return
