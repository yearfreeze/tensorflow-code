"""
using tensorflow as backend do a simple multiply
"""
import tensorflow as tf
sess=tf.Session()

x_val=5

a=tf.Variable(tf.constant(4.))
x=tf.placeholder(dtype=tf.float32)
mult=tf.multiply(a,x)

loss=tf.square(mult-50.)

init=tf.global_variables_initializer()
sess.run(init)

opt=tf.train.GradientDescentOptimizer(0.01)
train_step=opt.minimize(loss)

for i in range(10):
	sess.run(train_step,{x:x_val})
	a_val=sess.run(a)
	mult_output=sess.run(mult,{x:x_val})
	print(str(a_val)+'*'+str(x_val)+'='+str(mult_output))
