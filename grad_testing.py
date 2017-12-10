import tensorflow as tf
import numpy as np

x=tf.placeholder(dtype=tf.float32,shape=[None,4])
y=tf.placeholder(dtype=tf.float32,shape=[None,1])

A=tf.Variable(tf.random_normal(shape=[4,1]))
b=tf.Variable(tf.constant(0.))

score=tf.matmul(x,A)+b
prob=tf.nn.sigmoid(score)

loss=tf.reduce_mean((prob-y)**2)

#initializion of all variables
init=tf.global_variables_initializer()
s=tf.Session()
s.run(init)
#to compute gradients
my_opt=tf.train.GradientDescentOptimizer(0.2)
my_com=my_opt.compute_gradients(loss=loss,var_list=[A,b])
train=my_opt.apply_gradients(my_com)

#traing
xd=np.random.random((5,4))
yd=np.random.random((5,1))
print('A is'+str(s.run(A))+'\n')
print('prob is'+str(s.run(prob,{x:xd,y:yd}))+'loss is'+str(s.run(loss,{x:xd,y:yd}))+'\n')
print('compute_gradients'+str(s.run(my_com,{x:xd,y:yd}))+'\n')
"""
s.run(train,{x:xd,y:yd})#上一步计算梯度，这一步使用上一步计算出来的梯度更新权重 w-learning_rate*dw
print('A is'+str(s.run(A)))

s.run(train,{x:xd,y:yd})
print('A is'+str(s.run(A)))
"""
mczz=s.run(my_com,{x:xd,y:yd})
print('mczz0:'+str(mczz[0][0]))
ads=np.random.random((4,1))
mczz[0][0][0]+=1