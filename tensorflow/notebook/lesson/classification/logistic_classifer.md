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
$D(p,l)=-\sum_i({l_i}{\log(p_i)})$

As result,we summarize the logistic classifier:</br>
$x$ -> $Wx+b=y$ -> $Softmax(y)=p$ -> $D(p,l)$ -> $\frac{1}{N}\sum_iD$
</br>or:</br>
$loss=\frac{1}{N}\sum_iD(S(Wx_i+b)+l_i)$
</br>This expression represent **AVERAGE CROSS-ENTROPY**,data $x$ & $l$ are Tensor(matrix and vector usually)</br>
There are many mathod to make $loss$ smaller.The first is **GRADIENT DESCENT**.In this method, we use derivative of $loss(W)$.Such as:</br>
$-\alpha\nabla{loss(W_1,W_2,...W_n)}$
</br>
Where $\alpha$ is **learning rate**, $W_i$ is weights.

#varity of gradient descent
>1.batch gradient descent</br>
>2.stochastic gradient descent

The stochastic gradient descent is useful.And there are two method to extend it:</br>
>1.momentum : use mean of two nearby gradients intead of one gradient.</br>
>2.learning rate decay : decay learning rate with decrease of loss.</br>

So,there many parameter to set for SGD method:
>1.initial learning rate</br>
>2.learning rate decay</br>
>3.momentum</br>
>4.batch size</br>
>5.initial weight & bias</br>

And, there is a auto-method named ADAGRAD,it auto change 1.initial learning rate 2.learning rate decay 3.momentum,you just need care param 4. & 5.</br>
