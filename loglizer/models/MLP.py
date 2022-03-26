import numpy as np
from sklearn.neural_network import MLPClassifier
from ..utils import metrics

class MLP(object):

    def __init__(self, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, verbose=False):
        self.classifier = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
        self.verbose = verbose

    def fit(self, X, y):
        if self.verbose: 
            print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        if self.verbose:
            print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        if self.verbose:
            print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
