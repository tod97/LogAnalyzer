import numpy as np
from sklearn import neighbors
from ..utils import metrics

class NearestN(object):

    def __init__(self, n_neighbors=15, weights="uniform", verbose=False):
        self.classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
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
