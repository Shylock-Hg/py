import tensorflow as tf

#set Variable
W = tf.Variable(.3,dtype = tf.float32)
b = tf.Variable(-.3,dtype = tf.float32)

#set input and output
x = tf.placeholder(tf.float32)
y_ = W * x - b
y = tf.placeholder(tf.float32)

#loss
loss = tf.reduce_sum(tf.square(y_ - y))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#trainning data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

#train
init = tf.global_variables_initializer()
s = tf.Session()
s.run(init)
for i in range(1000):
    s.run(train,{x:x_train,y:y_train})

#evaluate
cw, cb, closs = s.run([W,b,loss], {x:x_train, y:y_train})
print('the currnet w is {}, the current b is {},the current loss is {}.'.format(cw, cb, closs))
