from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import graphviz
import random

def pipeline_tree_forest(X,y):
    # split data into train+validation set and test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)
    # split train+validation set into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split( X_trainval, y_trainval, random_state=1)
    print("Size of training set: {} size of validation set: {} size of test set: {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

    pipeline = Pipeline([
        ('clf',None)
    ])

    # Grid Searching to select the best parameters for the pipeline
    best_score = 0
    best_parameters = {}
    for classifier in [RandomForestClassifier(), ExtraTreesClassifier(), AdaBoostClassifier, GradientBoostingClassifier]:
        for n_estimators  in [10,50,100,150,200]:
            for max_depth in [5,10,15,20]:
                for oob_score in [True,False]:
                    for algorithm in ['SAMME', 'SAMME.R']:
                        for loss in ['deviance','exponential']:
                            for criterion in ['friedman_mse','mse','mae']:
                                parameters = {
                                    'clf': classifier,
                                    'clf__n_estimators': n_estimators,
                                    'clf__max_depth': max_depth,
                                    'clf__oob_score': oob_score
                                }

                                if isinstance(classifier, AdaBoostClassifier):
                                    del parameters['clf__max_depth']
                                    parameters['clf__algorithm'] = algorithm

                                if isinstance(classifier, GradientBoostingClassifier):
                                    del parameters['clf__oob_score']
                                    parameters['clf__loss'] = loss

                                pipeline.set_params(**parameters)

                                for param, value in parameters.items():
                                    print("-> Training {:} Classifier with {:} = {:}".format(classifier.__class__.__name__,param,value))
                                
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

def tree(X,y,should_viz=False):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=random.randint(1,50))
    clf = DecisionTreeClassifier()
    #clf = clf.fit(X_train,y_train)
    print("-> Training accuracy of the decision tree classifier: {:.2f}"
          .format(cross_val_score(clf,X,y).mean()))

    if should_viz:
        feature_names = X.columns.values.tolist()
        export_graphviz(clf,out_file="visualisations/bbb_tree.dot", class_names=["Pass","Not Pass"], feature_names=feature_names, impurity=False,filled=True)
        with open('visualisations/bbb_tree.dot') as f:
            dot_graph = f.read()
        gh = graphviz.Source(dot_graph,format='png')
        gh.render(filename="visualisations/bbb_tree_smd")

    return clf

def random_forest(X,y):
    # X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=random.randint(1,50))
    clf = RandomForestClassifier(n_estimators=20)
    # clf = clf.fit(X_train, y_train)
    print("-> Training accuracy of the random forest {:.2f}".format(cross_val_score(clf,X,y).mean()))

    return clf

def extra_trees(X,y):
    # X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=random.randint(1,50))
    clf = ExtraTreesClassifier(n_estimators=20)
    # clf = clf.fit(X_train, y_train)
    print("-> Training accuracy of the extra trees {:.2f}".format(cross_val_score(clf,X,y).mean()))
    return clf

def adaboost_trees(X,y):
    clf = AdaBoostClassifier(n_estimators=150,learning_rate=0.45)
    print("-> Training accuracy of the AdaBoost trees {:.2f}".format(cross_val_score(clf,X,y).mean()))

def gradientboost_trees(X,y):
    clf = GradientBoostingClassifier(n_estimators=150,learning_rate=0.45,max_depth=10)
    print("-> Training accuracy of the Gradient Boost trees {:.2f}"
          .format(cross_val_score(clf,X,y).mean()))
