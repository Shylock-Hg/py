'''
@brief: tht DNN classifier to class iris
@author: Shylock Hg
@time: 2017/8/17
@email: tcath2s@icloud.com
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

#downloaded in current dir
IRIS_TRAINING = 'iris_training.csv'
IRIS_TEST = 'iris_test.csv'

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename = IRIS_TRAINING,
        target_dtype = np.int,features_dtype = np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename = IRIS_TEST,
        target_dtype = np.int,features_dtype = np.float32)

feature_columns = [tf.contrib.layers.real_valued_column("",dimension = 4)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
        hidden_units = [10,20,10], n_classes = 3)


#fit the classifier
classifier.fit(x = training_set.data, y = training_set.target, steps = 2000)

#evaluate the classifier

accuracy = classifier.evaluate(x = test_set.data, y = test_set.target)['accuracy']
print('the accuracy is : {}'.format(accuracy))

#classifiy new sample
def new_sample():
    return np.array([[6.4,3.2,4.5,1.5],[5.8,3.1,5.0,1.7]], dtype = np.float32)
predictions = list(classifier.predict(new_sample))
print('the new sample predicts is : {}'.format(predictions))
