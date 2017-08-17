'''
@brief: demo of startion of tensorflow
@author: TcAth2s
@time: 2017/8/13
@email: tcath2s@icloud.com
'''

import tensorflow as tf


'''
constant addition
'''
nd1 = tf.constant(1.,dtype = tf.float32)
nd2 = tf.constant(3.,dtype = tf.float32)

op1 = tf.add(nd1,nd2)

s = tf.Session()

print(s.run(op1))

'''
placeholder
'''
h1 = tf.placeholder(tf.float32)
h2 = tf.placeholder(tf.float32)



op2 = h1+h2

print(s.run(op2,{h1:[1.,2.],h2:[5.,8.]}))

op3 =  op2 * 3

print(s.run(op3,{h1:[2.,5.],h2:[6.,9.]}))


'''
linear equation
'''
w = tf.Variable(.3,dtype = tf.float32)
b = tf.Variable(-.3,dtype = tf.float32)
x = tf.placeholder(tf.float32)

y = w * x + b

init = tf.global_variables_initializer()
s.run(init)

print(s.run(y,{x:[1,2,3,4]}))

y_ = tf.placeholder(tf.float32)
squared_delta = tf.square(y - y_)
loss = tf.reduce_sum(squared_delta)
print(s.run(loss,{x = [1,2,3,4],y_ = [0,-1,-2,-3]}))

'''
train
'''
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

s.run(init)
for i in range(1000):
    s.run(train,{x:[1,2,3,4],y_:[0,-1,-2,-3]})
print(s.run([w,b]))
