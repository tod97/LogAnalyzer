from loglizer.models import PCA, SVM
from loglizer import dataloader, preprocessing

struct_log = './loghub/BGL/BGL_2k.log_structured.csv' # The structured log file

(x_train, _), (_, _), _ = dataloader.load_BGL(struct_log, window='session',  split_type='sequential', save_csv=True)

""" README
!! the element inside x_train is the event sequence
!! in the 2k example, there are 12 different events
datas are composed like this:
    x_train = [list(['e3df2680']) list(['e3df2680']) list(['dba996ef']) ...]
"""

feature_extractor = preprocessing.FeatureExtractor()
x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', normalization='zero-mean')
""" README
we obtain x_train using tf-idf and zero-mean to normalize datas:

x_train =  [[ 2.17737193 -0.26591861 -0.08536254 ... -0.0076379  -0.0076379 -0.0076379 ]
            [-0.21827615  0.12226143 -0.08536254 ... -0.0076379  -0.0076379 -0.0076379 ]
            [-0.21827615  0.12226143 -0.08536254 ... -0.0076379  -0.0076379 -0.0076379 ]
"""

model = PCA()
model.fit(x_train)