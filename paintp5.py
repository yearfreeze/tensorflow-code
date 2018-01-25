# -*- coding: utf-8 -*-	
"""
Created on Sat Jan 24 12:17:25 2018

@author: freeze
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf

start_position=300
fold_length=4
n_step=15
batch_size=30
n_input_length=fold_length-1
learning_rate=0.01
hidden_size=10
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

def copy_data(data):#function to form designed matrix x&y
    choice=np.random.choice(len(data)-n_step,size=batch_size,replace=False)
    design_x=np.zeros(shape=(batch_size,n_step,n_input_length))
    design_y=np.zeros(shape=(batch_size*n_step,1))
    num=0
    for c in choice:
        design_x[num]=data[c:c+n_step,0:n_input_length]
        num+=1
    num=0
    for c in choice:
        for j in range(c,c+n_step):
            design_y[num]=data[j,n_input_length]
            num+=1
    
    return design_x,design_y
	


def normalize_cols(m):
    col_max=m.max(axis=0)
    col_min=m.min(axis=0)
    return (m-col_min)/(col_max-col_min)
	
def fix(mat,digit):
	row=mat.shape[0]
	col=mat.shape[1]
	mc=np.zeros_like(mat)
	mc[0:row-1]=mat[1:]
	mc[row-1,0:col-1]=mat[row-1,1:col]
	mc[row-1,col-1]=digit
	return mc
    
def reverse(L,ma,mi):
	emp=[]
	for i in L:
		i=i*(ma-mi)+mi
		emp.append(i)
	return emp

matfn='P5.mat'
data=sio.loadmat(matfn)
raw_data=-data['P5']  #positive data
fold_data=generate(raw_data,fold_length)#choosing 

max_element=raw_data.max()
min_element=raw_data.min()

nor_data=normalize_cols(fold_data)

#reverse_data=nor_data*(max_element-min_element)+min_element

#designing my simple rnn net
x=tf.placeholder(shape=(None,n_step,n_input_length),dtype=tf.float32)
y=tf.placeholder(shape=(None,1),dtype=tf.float32)
w=tf.Variable(tf.random_normal(shape=[hidden_size,1]))
b=tf.Variable(tf.random_normal(shape=[1,]))

rnn_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_size)
init_state=rnn_cell.zero_state(batch_size,dtype=tf.float32)
outputs,state=tf.nn.dynamic_rnn(rnn_cell,x,initial_state=init_state,dtype=tf.float32)
reou=tf.reshape(outputs,[-1,hidden_size])
pred_y=tf.matmul(reou,w)+b

loss=tf.reduce_mean((pred_y-y)**2)
my_opt=tf.train.AdamOptimizer(learning_rate)
train=my_opt.minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

lossing=[]
for i in range(iteration):
	[d_x,d_y]=copy_data(nor_data)
	sess.run(train,{x:d_x,y:d_y})
	lose=sess.run(loss,{x:d_x,y:d_y})
	lossing.append(lose)
	print('generation:'+str(i)+' loss is'+str(lose))
	
'''
[px,py]=copy_data(nor_data)

print(sess.run(pred_y,{x:px}))
print('-----------')
print('py is'+str(py))
'''
fitting_answer=[]
init_vector=np.zeros((batch_size,n_step,n_input_length))
init_vector[-1]=nor_data[start_position-n_step:start_position,0:n_input_length]
#reshape
#print(init_vector)
time=start_position
while time<800:
	#print(init_vector)
	#print('------------------------')
	ans=sess.run(pred_y,{x:init_vector})
	#print(ans.shape)
	print(ans[-1])
	fitting_answer.append(ans[-1][0])
	temp=fix(init_vector[-1],ans[-1][0])
	init_vector[-1]=temp
	#print(fitting_answer)
	time+=1
#print(fitting_answer) 


fuck_answer=reverse(fitting_answer,max_element,min_element)
#print(fuck_answer)
plt.plot(range(800-start_position),raw_data[start_position:800],'b-')
plt.plot(range(800-start_position),fuck_answer,'r-')
plt.show()