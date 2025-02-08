# Arquivo com todas as funcoes e codigos referentes aos experimentos

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV



def logistic_regression_train(X_train, y_train):
    """
    Função que treina um modelo de Regressão Logística com GridSearchCV
    """
    model = LogisticRegression(max_iter=1000)
    param_grid = {
        'C': [0.1],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def naive_bayes_train(X_train, y_train):
    """
    Função que treina um modelo de Naive Bayes
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def decision_tree_train(X_train, y_train):
    """
    Função que treina um modelo de Árvore de Decisão com GridSearchCV
    """
    model = DecisionTreeClassifier()
    param_grid = {
        'criterion': ['entropy'],
        'max_depth': [5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 4]
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)
    return grid.best_estimator_




def random_forest_train(X_train, y_train):
    """
    Função que treina um modelo de Random Forest com GridSearchCV
    """
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [300],
        'max_depth': [None],
        'min_samples_split': [8],
        'min_samples_leaf': [3],
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)
    return grid.best_estimator_




def svm_train(X_train, y_train):
    """
    Função que treina um modelo de SVM
    """
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    return model



def knn_train(X_train, y_train):
    """
    Função que treina um modelo de KNN com GridSearchCV
    """
    model = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [7],
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)
    return grid.best_estimator_



def mlp_train(X_train, y_train):
    """
    Função que treina um modelo de MLP com GridSearchCV
    """
    model = MLPClassifier(max_iter=1000)
    param_grid = {
        'activation': ['logistic'],
        'alpha': [0.01],
        'learning_rate': ['adaptive']
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)
    return grid.best_estimator_



