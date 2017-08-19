#definition
Just linear classifier:</br>
$Wx+b = y$
>W: weights</br>
>b: bias</br>
In machine learning, we train model to make **Parameters** of model fit the function(Map data to label).

But the output y is not useful directly.Usually, we use softmax function transfer 'y' to probablility.Softmax is :</br>
$S(y_i)=(\frac{e^{y_i}}{\sum_i{e^{y_j}}})$
</br>After transfering,the bigger 'y' corresponds bigger probability.
```sequence
LOGITS SCORE -> PROBABILITIES : Softmax
```

>note: if scalar up 'y', the probability will closer to 0 or 1,means the stdev will become bigger.</br>
note: if scalar down 'y', the probability will closer to each other,means the stdev will become smaller.</br>

Finally,we use one-hot vector represent result of classification.

But,when the one-hot vector is too large,this expression will become inefficient.So we use cross-entropy represent the loss between predict output and data.
Cross Entropy:</br>
$D(S,L)=-\sum_i({L_i}{\log(S_i)})$

As result,we summarize the logistic classifier:</br>
$x$ -> $Wx+b=y$ -> $Softmax(y)=S$ -> $D(S,L)$
