from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
import graphviz
import random

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
