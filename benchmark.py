import sys
sys.path.append('../')
import pandas as pd
from loglizer.models import *
from loglizer import dataloader, preprocessing

struct_log = './loghub/HDFS/HDFS_100m.log_structured.csv' # The benchmark dataset
label_file = './loghub/HDFS/anomaly_label.csv' # The anomaly label file
feature_extractor = preprocessing.FeatureExtractor()
(x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log, label_file=label_file, window='session', train_ratio=0.5, split_type='uniform')
benchmark_results = []
train_results = []
test_results = []

def model_evaluate(_model, model, x_te, y_test, x_train, y_train):
    x_test = feature_extractor.transform(x_te)
    precision, recall, f1 = model.evaluate(x_train, y_train)
    benchmark_results.append([_model + '-train', precision, recall, f1])
    train_results.append([_model, precision, recall, f1])
    precision, recall, f1 = model.evaluate(x_test, y_test)
    benchmark_results.append([_model + '-test', precision, recall, f1])
    test_results.append([_model, precision, recall, f1])


#_model = 'DecisionTree'
x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
model = DecisionTree()
model.fit(x_train, y_train)
model_evaluate('DecisionTree', model, x_te, y_test, x_train, y_train)

#_model = 'InvariantsMiner'
x_train = feature_extractor.fit_transform(x_tr)
model = InvariantsMiner(epsilon=0.5)
model.fit(x_train)
model_evaluate('InvariantsMiner', model, x_te, y_test, x_train, y_train)

#_model = 'IsolationForest'
x_train = feature_extractor.fit_transform(x_tr)
model = IsolationForest(random_state=2019, max_samples=0.9999, contamination=0.03, n_jobs=4)
model.fit(x_train)
model_evaluate('IsolationForest', model, x_te, y_test, x_train, y_train)

#_model = 'LogClustering'
x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
model.fit(x_train[y_train == 0, :]) # Use only normal samples for training
model_evaluate('LogClustering', model, x_te, y_test, x_train, y_train)

#_model = 'LR'
x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
model = LR()
model.fit(x_train, y_train)
model_evaluate('LR', model, x_te, y_test, x_train, y_train)

#_model = 'PCA'
x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', normalization='zero-mean')
model = PCA()
model.fit(x_train)
model_evaluate('PCA', model, x_te, y_test, x_train, y_train)

#_model = 'SVM'
x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
model = SVM()
model.fit(x_train, y_train)
model_evaluate('SVM', model, x_te, y_test, x_train, y_train)

#_model = 'MLP'
x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
model = MLP()
model.fit(x_train, y_train)
model_evaluate('MLP', model, x_te, y_test, x_train, y_train)

#_model = 'MLP'
x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
model = NearestN(n_neighbors=100)
model.fit(x_train, y_train)
model_evaluate('Nearest Neighbors', model, x_te, y_test, x_train, y_train)

print('====== Train Results ======')
print(pd.DataFrame(train_results, columns=['Model', 'Precision', 'Recall', 'F1']))
print('\n====== Test Results ======')
print(pd.DataFrame(test_results, columns=['Model', 'Precision', 'Recall', 'F1']))
#pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']).to_csv('benchmark_result.csv', index=False)