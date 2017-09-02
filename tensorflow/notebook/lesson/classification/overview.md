#definition
Map data(Tensor) to label.</br>Classifier is just the model of the Map function.</br>Classifier input data output corresponding label.</br>

```sequence
Data -> Label : Model
```
#Train
There are some problem in training model.</br>
>1.You should divide data to three(train,validation,eval),the eval data is use once to eval perfermance of model.In this case , we can improve prediction ability of model.We should care generalization of Model firstly, then fit training data as possibly.
