# @Author: AJ Ibraheem <AJ>
# @Date:   2016-11-12T12:23:30+00:00
# @Email:  ibraheemaj@icloud.com
# @Last modified by:   AJ
# @Last modified time: 2016-11-12T12:23:35+00:00

from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import matplotlib.pyplot as plt

def pipeline_svm(X,y,isFingerprint=False):
    # split data into train+validation set and test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)
    # split train+validation set into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split( X_trainval, y_trainval, random_state=1)
    print("Size of training set: {} size of validation set: {} size of test set: {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

    pipeline = None
    feature_union = FeatureUnion([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=50)) #TODO: GridSearch n_components parameter later
        ])
    if not isFingerprint:
        pipeline = Pipeline([
            ('preprocessors', feature_union),
            ("svm", SVC(kernel='linear',probability=True,class_weight='balanced'))
        ])
    else:
        pipeline = Pipeline([("svm", SVC(kernel='linear',probability=True,class_weight='balanced'))])

    # Debugging purposes only 
    # for param,value in pipeline.get_params().items():
    #     print("Parameter = {:} \t Value= {:}".format(param,value))

    # Grid Searching to select the best parameters for the pipeline
    best_score = 0
    best_parameters = {}
    for gamma in [0.01, 1, 100]: #[0.001, 0.01, 0.1, 1, 10, 100]
        for C in [0.01, 1, 100]:
            for pca_n_components in [30,50,80]:
                for kernel in ['linear','rbf','poly','sigmoid']:
                    parameters = {
                        'svm__kernel': kernel,
                        'svm__C': C,
                        'svm__gamma': gamma
                    }
                    if not isFingerprint:
                        parameters['preprocessors__pca__n_components'] = pca_n_components
                    
                    pipeline.set_params(**parameters)

                    for param, value in parameters.items():
                        print("-> Training SVM Classifier with {:} = {:}".format(param,value))

                    pipeline.fit(X_train, y_train)
                    #TODO: Update to use cross validation into_ml_pg_264, although it increases train time
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

    plt.savefig(viz_name,format='png',dpi=500)
    return
