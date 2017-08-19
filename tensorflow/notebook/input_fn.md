#Building Input Functions with tf.estimator
##1.Custom Input Pipelines with input_fn
The input_fn is used to pass feature and target data to the train, evaluate, and predict methods of the Estimator. The user can do feature engineering or pre-processing inside the input_fn. Here's an example taken from the tf.estimator Quickstart tutorial:</br>
```python
import numpy as np

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

classifier.train(input_fn=train_input_fn, steps=2000)
```

###1.1Anatomy of an input_fn
```python
def my_input_fn():

    # Preprocess your data here...

    # ...then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return feature_cols, labels
```
###feature_cols
A dict containing key/value pairs that map feature column names to Tensors (or SparseTensors) containing the corresponding feature data.
###labels
A Tensor containing your label (target) values: the values your model aims to predict.
###1.2Converting Feature Data to Tensors
####1.2.1np.array & pd.frame convert to Tensors
If your feature/label data is a python array or stored in pandas dataframes or numpy arrays, you can use the following methods to construct input_fn:</br>
**numpy.array**
```python
import numpy as np
# numpy input_fn.
my_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_data)},
    y=np.array(y_data),
    ...)
```
**pandas.dataframe**
```python
import pandas as pd
# pandas input_fn.
my_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({"x": x_data}),
    y=pd.Series(y_data),
    ...)
```
####1.2.2for sparse,categorical data
```python

sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],
                                values=[6, 0.5],
                                dense_shape=[3, 5])
```
>1.the **indices** is the indices who specify the values</br>
>2.the **vlaues** is the values to specify to **indices** elements</br>
>3.the **dense_shape** is the shape of generated Tensors</br>

###1.3Passing input_fn Data to Your Model
Transfer data to train model:
```python
#the **my_input_fn** is a function
classifier.fit(input_fn = my_input_fn, steps = 2000)
```
The method to transfer params to input_fn
>the method 1 - wrap a parametered function
```python
def my_input_fn(data_set):
  ...

def my_input_fn_training_set():
  return my_input_fn(training_set)

classifier.train(input_fn=my_input_fn_training_set, steps=2000)
```
>the method 2 - with **functools.partial**
```python
classifier.train(
    input_fn=functools.partial(my_input_fn, data_set=training_set),
    steps=2000)
```
>the method 3 - wraped by **lambda** expression
```python

classifier.evaluate(input_fn=lambda: my_input_fn(test_set), steps=2000)
```
You can reuse the input_fn with parametered such as:
```python

classifier.evaluate(input_fn=lambda: my_input_fn(test_set), steps=2000)
```
>In the above demo,you can transfer training,test&predict data with **one input_fn**
###1.4A Neural Network Model for Boston House Values
>Boston House Data:</br>


|   feature  |  means |
 ----------------- | ----------------------
| CRIM | Crime rate per capita  |
| ZN  | Fraction of residential land zoned to permit 25,000+ sq ft lots   |
| INDUS  | Fraction of land that is non-retail business |
| NOX | Concentration of nitric oxides in parts per 10 million |
| RM | Average Rooms per dwelling |
| AGE | Fraction of owner-occupied residences built before 1940|
| DIS | Distance to Boston-area employment centers|
| TAX | Property tax rate per $10,000 |
| PTRATIO | Student-teacher ratio |

##2Setup
###2.1Importing the Housing Data
>import the used models
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
```
>import the data by pandas
```python

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
```
>skipinitialspace = True : skip the sapce in elements</br>
>skiprows = 1 : skip the first raw in csv
###2.2Defining FeatureColumns and Creating the Regressor
Next, create a list of FeatureColumns for the input data, which formally specify the set of features to use for training. Because all features in the housing data set contain continuous values, you can create their FeatureColumns using the tf.contrib.layers.real_valued_column() function:</br>
```python
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
```
construct the DNNRegressor:</br>
```python
regressor = tf.estimator.DNNRegressor(feature_columns = feature_cols,
        hidden_units = [10,10], model_dir = '/tmp/regressor')
```
###2.3Building the input_fn
To pass input data into the regressor, write a factory method that accepts a pandas Dataframe and returns an input_fn:</br>
```python

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y = pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)
```
>**num_epochs**: controls the number of epochs to iterate over data. For training, set this to **None**, so the
input_fn keeps returning data until the required number of train steps is reached. For evaluate and predict, set this to **1**, so the input_fn will iterate over the data once and then raise OutOfRangeError. That error will signal the Estimator to stop evaluate or predict.</br>
>**shuffle**: Whether to shuffle the data. For evaluate >and predict, set this to **False**, so the input_fn >iterates over the data sequentially. For train, set this >to **True**.</br>
###2.4Training the Regressor
To train the neural network regressor, run train with the training_set passed to the input_fn as follows:
```python
regressor.fit(input_fn = get_input_fn(training_set),steps = 5000)
```
###2.5Evaluating the Model
Next, see how the trained model performs against the test data set. Run evaluate, and this time pass the test_set to the input_fn:</br>
```python
eval = regressor.evaluate(input_fn = get_input_fn(test_set,num_epochs = 1,shuffle = False))
#print loss
print('the loss is : {}'.format(eval['loss']))
```
###2.6Making Predictions
Finally, you can use the model to predict median house values for the prediction_set, which contains feature data but no labels for six examples:</br>
```python
y = regressor.predict(input_fn = get_input_fn(prediction_set,num_epochs = 1,shuffle = False))
#.predict func return a iterator of dicts,convert to list and print preictions
predictions = list(p['predictions'] for p in itertools.slice(y,6))
print('the predictions is : {}'.format(predictions))
```
