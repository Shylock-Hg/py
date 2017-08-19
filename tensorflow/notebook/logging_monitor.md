#Logging and Monitoring Basics with tf.contrib.learn
##1overview
Fortunately, tf.contrib.learn offers another solution: a Monitor API designed to help you log metrics and evaluate your model while training is in progress.In the following sections, you'll learn how to enable logging in TensorFlow, set up a ValidationMonitor to do streaming evaluations, and visualize your metrics using TensorBoard.
##2Enabling Logging with TensorFlow
TensorFlow uses five different levels for log messages. In order of ascending severity, they are DEBUG, INFO, WARN, ERROR, and FATAL.</br>
You will get the logging that **above** specified level.And the default level is WARN,but INFO is more useful as it provide information about fit progress.</br>
Add the following line to the beginning of your code (right after your imports):</br>
```python
tf.logging.set_verbosity(tf.logging.INFO)
```
##3Configuring a ValidationMonitor for Streaming Evaluation
| Monitor | Description |
---------|----------------
| CaptureVariable | Saves a specified variable's values into a collection at every n steps of training |
| PrintTensor | Logs a specified tensor's values at every n steps of training |
| SummarySaver | Saves tf.Summary protocol buffers for a given tensor using a tf.summary.FileWriter at every n steps of training |
| ValidationMonitor | Logs a specified set of evaluation metrics at every n steps of training, and, if desired, implements early stopping under certain conditions |

###3.1Evaluating Every N Steps
>1.In above code, we get logging about **fit** every 50 train steps</br>
>2.Place this code right before the line instantiating the classifier.</br>
```python

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50)
```
>3.ValidationMonitors rely on saved checkpoints to perform evaluation operations, so you'll want to modify instantiation of the classifier to add a tf.contrib.learn.RunConfig.</br>
```python
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=3,
    model_dir="/tmp/iris_model",
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
```
>4Finally, to attach your validation_monitor, update the fit call to include a monitors param, which takes a list of all monitors to run during model training:</br>
```python

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
               monitors=[validation_monitor])
```
###3.2Customizing the Evaluation Metrics with MetricSpec
By default, if no evaluation metrics are specified, ValidationMonitor will log both loss and accuracy, but you can customize the list of metrics that will be run every 50 steps. To specify the exact metrics you'd like to run in each evaluation pass, you can add a metrics param to the ValidationMonitor constructor. metrics takes a dict of key/value pairs, where each key is the name you'd like logged for the metric, and the corresponding value is a MetricSpec object.</br>
The MetricSpec constructor accepts four parameters:</br>
>1.**metric_fn**: The function that calculates and returns the value of a metric. This can be a predefined function available in the **tf.contrib.metrics** module, such as:</br>
`tf.contrib.metrics.streaming_precision` or `tf.contrib.metrics.streaming_recall`</br>
Alternatively, you can define your own custom metric function, which must take predictions and labels tensors as arguments (a weights argument can also optionally be supplied). The function must return the value of the metric in one of two formats:</br>
`a single Tensor` or `A pair of ops (value_op, update_op), where value_op returns the metric value and update_op performs a corresponding operation to update internal model state.`</br>
2.**prediction_key**: The key of the tensor containing the predictions returned by the model. This argument may be omitted if the model returns either a single tensor or a dict with a single entry. For a DNNClassifier model, class predictions will be returned in a tensor with the key :</br>
`tf.contrib.learn.PredictionKey.CLASSES`</br>
3.**label_key**: The key of the tensor containing the labels returned by the model, as specified by the model's input_fn. As with prediction_key, this argument may be omitted if the input_fn returns either a single tensor or a dict with a single entry. In the iris example in this tutorial, the DNNClassifier does not have an input_fn (x,y data is passed directly to fit), so it's not necessary to provide a label_key.</br>
4.**weights_key**: Optional. The key of the tensor (returned by the input_fn) containing weights inputs for the metric_fn.</br>

There is a example of **validation_metrics**:</br>
```python

validation_metrics = {
    'accuracy' :
        tf.contrib.learn.MetricSpec(
            metric_fn = tf.contrib.metrics.streaming_accuracy,
            prediction_key = tf.contrib.learn.PredictionKey.CLASSES
        ),
    'precision' :
        tf.contrib.learn.MetricSpec(
            metric_fn = tf.contrib.metrics.streaming_precision,
            prediction_key = tf.contrib.learn.PredictionKey.CLASSES
        ),
    'recall' :
        tf.contrib.learn.MetricSpec(
            metric_fn = tf.contrib.metrics.streaming_recall,
            prediction_key = tf.contrib.learn.PredictionKey.CLASSES
        )
}
```
Add **validation_metrics** to **monitor**:</br>
```python

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics)
```
###3.3Early Stopping with ValidationMonitor
In addition to logging eval metrics, ValidationMonitors make it easy to implement early stopping when specified conditions are met, via three params:</br>
| PARAM | Description |
------- | -------------
| early_stopping_metric | Metric that triggers early stopping (e.g., loss or accuracy) under conditions specified in early_stopping_rounds and early_stopping_metric_minimize. Default is "loss".|
| early_stopping_metric_minimize | True if desired model behavior is to minimize the value of early_stopping_metric; False if desired model behavior is to maximize the value of early_stopping_metric. Default is True.|
| early_stopping_rounds | Sets a number of steps during which if the early_stopping_metric does not decrease (if early_stopping_metric_minimize is True) or increase (if early_stopping_metric_minimize is False), training will be stopped. Default is None, which means early stopping will never occur. |

There is a example about the three parameters:</br>
```python
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)
```
##4.Visualizing Log Data with TensorBoard
Reading through the log produced by ValidationMonitor provides plenty of raw data on model performance during training, but it may also be helpful to see visualizations of this data to get further insight into trends—for example, how accuracy is changing over step count. You can use TensorBoard (a separate program packaged with TensorFlow) to plot graphs like this by setting the logdir command-line argument to the directory where you saved your model training data (here, /tmp/iris_model). Run the following on your command line:</br>
```shell
$ tensorboard --logdir=/tmp/iris_model/
Starting TensorBoard 39 on port 6006
```
Then navigate to http://0.0.0.0:<port_number> in your browser, where <port_number> is the port specified in the command-line output (here, 6006).
