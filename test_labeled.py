from loglizer.models import PCA, SVM
from loglizer import dataloader, preprocessing
import matplotlib.pyplot as plt
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

struct_log = './loghub/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = './loghub/HDFS/anomaly_label.csv' # The anomaly label file

(x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log, label_file=label_file, window='session',  train_ratio=0.5, split_type='uniform')

""" README
!! the element inside x_train is the event sequence
!! in the 2k example, there are 12 different events
datas are composed like this:
    x_train = [list(['e3df2680']) list(['e3df2680']) list(['dba996ef']) ...]
    y_train = [0 0 0 0 1 0 0 ...]
"""


feature_extractor = preprocessing.FeatureExtractor()
x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', normalization='zero-mean')

""" README
we obtain x_train using tf-idf and zero-mean to normalize datas:

x_train =  [[ 2.17737193 -0.26591861 -0.08536254 ... -0.0076379  -0.0076379 -0.0076379 ]
            [-0.21827615  0.12226143 -0.08536254 ... -0.0076379  -0.0076379 -0.0076379 ]
            [-0.21827615  0.12226143 -0.08536254 ... -0.0076379  -0.0076379 -0.0076379 ]
"""

x_test = feature_extractor.transform(x_test)
""" README
transform execute the same things as fit_transform with default parameters
"""

model = PCA()
model.fit(x_train)

#model.evaluate(x_train, y_train)