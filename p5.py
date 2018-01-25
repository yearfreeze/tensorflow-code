# -*- coding: utf-8 -*-	
"""
Created on Sat Jan 20 20:17:25 2018

@author: freeze
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#paramter seting
start_position=300
fold_length=4
learning_rate=0.05
batch_size=25
hidden_size=6
iteration=1000

def generate(raw_data,fold_len):
    final_data=[]
    temp_data=[]
    for i in range(raw_data.shape[0]-fold_len):
        for j in range(i,i+fold_len):
            temp_data.append(raw_data[j])
        final_data.append(temp_data)
        temp_data=[]
    #return final_data
    final_data=np.array(final_data)
    return final_data.reshape(raw_data.shape[0]-fold_len,fold_len)

def normalize_cols(m):
    col_max=m.max(axis=0)
    col_min=m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

def diff(m):#diff the raw data using the next sub now one
    L=[]
    for i in range(len(m)-1):
        temp=m[i+1]-m[i]
        L.append(temp)
    return np.array(L).reshape(799,1)

def rediff(L,ma,mi):
	emp=[]
	for i in L:
		i=i*(ma-mi)+mi
		emp.append(i)
	return emp
	
def adding(L):
	add_L=[]
	temp=0
	for i in range(len(L)):
		temp+=L[i]
		add_L.append(temp)
	return add_L

#load data
matpn=u'P5.mat'
data=sio.loadmat(matpn)

raw_data=-data['P5']
diff_data=diff(raw_data)

max_element=(diff_data).max()
min_element=(diff_data).min()

nor_data=normalize_cols(diff_data)
no_data=generate(nor_data,fold_length)
train_data=no_data[0:start_position]



x=tf.placeholder(shape=[None,fold_length-1],dtype=tf.float32)
y=tf.placeholder(shape=[None,1],dtype=tf.float32)

w1=tf.Variable(tf.random_normal(shape=[fold_length-1,hidden_size]))
b1=tf.Variable(tf.random_normal(shape=[hidden_size]))
w2=tf.Variable(tf.random_normal(shape=[hidden_size,fold_length]))
b2=tf.Variable(tf.random_normal(shape=[fold_length]))
w3=tf.Variable(tf.random_normal(shape=[fold_length,1]))
b3=tf.Variable(tf.random_normal(shape=[1]))

layer1=tf.nn.sigmoid(tf.matmul(x,w1)+b1)
layer2=tf.nn.sigmoid(tf.matmul(layer1,w2)+b2)
y_target=(tf.matmul(layer2,w3)+b3)

loss=tf.reduce_mean((y_target-y)**2)
my_opt=tf.train.AdamOptimizer(learning_rate)
train=my_opt.minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

lossing=[]
for i in range(iteration):
	rand_index=np.random.choice(len(train_data),size=batch_size,replace=False)
	rand_x=train_data[rand_index,0:fold_length-1]
	rand_y=train_data[rand_index,fold_length-1].reshape(batch_size,1)
	sess.run(train,{x:rand_x,y:rand_y})
	lose=sess.run(loss,{x:rand_x,y:rand_y})
	lossing.append(lose)
	print('generation:'+str(i)+' loss is'+str(lose))
'''
print('-----------test-------------')
content_x=k[0:3,0:3]
content_y=k[0:3,3]
temp_x=train_data[0:3,0:3]
temp_y=train_data[0:3,3].reshape(3,1)
print(temp_x)
print (content_x)
answer=sess.run(y_target,{x:temp_x,y:temp_y})
answer=answer*(max_element-min_element)+min_element
print(answer)
print(content_y)
	
plt.plot(range(iteration),lossing,'k-')
plt.show()
'''
predict_point=[]
#print(len(train_data))#299-800
print(train_data[299])
#tt=train_data[299]
#print(tt[0:3])

initiaize_vector=list(train_data[start_position-1])
i=start_position-1
#print(type(initiaize_vector))
while i<800:
	prepare_x=np.array(initiaize_vector[0:fold_length-1])
	prepare_y=np.array(initiaize_vector[fold_length-1])
	px=prepare_x.reshape(1,fold_length-1)
	py=prepare_y.reshape(1,1)
	
	answer=sess.run(y_target,{x:px,y:py})
	
	print('i= '+str(i)+'list is: '+str(initiaize_vector))
	print('predict is '+str(answer[0][0]))
	
	predict_point.append(answer[0][0])
	initiaize_vector.pop(0)
	initiaize_vector.append(answer[0][0])
	i+=1

#print(predict_point)
rediff_predict=rediff(predict_point,max_element,min_element)
basement_add=adding(rediff_predict)
answers=raw_data[start_position+fold_length-2]+basement_add
print(answers)

plt.plot(range(800-start_position),raw_data[start_position:800],'b-')
plt.plot(range(800-start_position+1),answers,'r-')
plt.show()