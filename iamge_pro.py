# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:43:36 2017

the purpose of this code is testing double direction recurrence reual netword

@author: freeze
"""
import time
import tensorflow as tf
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')#绘画标记，否
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig 

from tensorflow.examples.tutorials.mnist import input_data

import wx 
 
#IO
mnist=input_data.read_data_sets('MNIST_data')

#parameter setings
learning_rate=0.01
max_samples=20000
batch_size=128
display_step=10

n_input=28
n_steps=28
n_hidden=256
n_classes=10

#define bidirectional LSTM
def BIRNN(x,weights,biases):
	x=tf.transpose(x,[1,0,2])
	x=tf.reshape(x,[-1,n_input])
	x=tf.split(x,n_steps)
	
	lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
	lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
	
	outputs,a,b=tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
	return tf.matmul(outputs[-1],weights)+biases

def generate_one_hot(label,high,width):
	g=np.zeros((high,width))
	l=len(label)
	for i in range(l):
		g[i][label[i]]=1
	return g

def random_down_index(cnt):
	a=random.randint(0,4000)
	while(mnist.test.labels[a]!=cnt):
		a=random.randint(0,4000)
	return a

x=tf.placeholder(dtype=tf.float32,shape=[None,28,28])
y=tf.placeholder(dtype=tf.float32,shape=[None,n_classes])

weights=tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
biases=tf.Variable(tf.random_normal([n_classes]))
pred=BIRNN(x,weights,biases)

pred_number=tf.argmax(pred,1)

lost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

class Example(wx.Frame):
	def __init__(self, parent, title):
		super(Example, self).__init__(parent, title = title, size = (800,500))
		self.trainTrigger=False
		#self.InitTensorflow()
		self.InitUI()
		self.Centre()
		self.Show()

	def InitUI(self): 
		p=wx.Panel(self) 
		whole_box=wx.BoxSizer(wx.HORIZONTAL)
		#set train static box
		train_box=wx.StaticBox(p,-1,'Train:')
		train_box_Sizer=wx.StaticBoxSizer(train_box,wx.VERTICAL)
		pbox=wx.BoxSizer(wx.VERTICAL)
		
		self.tb=wx.Button(p,-1,'trainButton')
		self.tb.Bind(wx.EVT_BUTTON,self.OnClick)  #setting dealling function 
		self.tx=wx.TextCtrl(p, size=(200,300), style=wx.TE_MULTILINE)
		self.tg=wx.Gauge(p, range = 15, size = (250, 25), style =  wx.GA_HORIZONTAL  ) #|wx.GA_TEXT
		
		pbox.Add(self.tb, 0, wx.ALL | wx.ALIGN_CENTER, 5)
		pbox.Add(self.tx, 0, wx.EXPAND | wx.CENTER, 5) 
		pbox.Add(self.tg, 0, wx.ALL | wx.CENTER, 5) 
		
		train_box_Sizer.Add(pbox, 0, wx.EXPAND|wx.CENTER, 10) 
		
		#set show static box
		show_box=wx.StaticBox(p,-1,'Show:')
		show_box_Sizer=wx.StaticBoxSizer(show_box,wx.VERTICAL)
		qbox=wx.BoxSizer(wx.VERTICAL)
		
		temp=wx.Image("blank.jpg",wx.BITMAP_TYPE_JPEG).Scale(200,300).ConvertToBitmap()
		self.bmp=wx.StaticBitmap(p,bitmap=temp)
		
		qbox.Add(self.bmp,0,wx.EXPAND|wx.ALIGN_CENTER, 5)
		show_box_Sizer.Add(qbox, 0, wx.ALL|wx.CENTER, 10)
		#set test static box
		test_box=wx.StaticBox(p,-1,'Test:')
		test_box_Sizer=wx.StaticBoxSizer(test_box,wx.VERTICAL)
		rbox=wx.BoxSizer(wx.VERTICAL)
		#compoent setting
		L=['0','1','2','3','4','5','6','7','8','9']
		self.combo=wx.ComboBox(p,choices=L)
		self.combo.Bind(wx.EVT_COMBOBOX,self.Oncombo)#setting dealling function 
		self.cbmp=wx.StaticBitmap(p,bitmap=temp)
		self.st=wx.StaticText(p,-1,style = wx.ALIGN_CENTER)
		self.st.SetLabel('pred: ')
		
		rbox.Add(self.combo, 0, wx.EXPAND|wx.ALIGN_CENTER, 5)
		rbox.Add(self.cbmp, 0, wx.ALL|wx.CENTER, 5) 
		rbox.Add(self.st, 0, wx.ALL|wx.CENTER, 5) 
		
		test_box_Sizer.Add(rbox, 0, wx.ALL|wx.CENTER, 10)
		whole_box.Add(train_box_Sizer,0, wx.EXPAND|wx.CENTER, 2)
		whole_box.Add(show_box_Sizer,0, wx.EXPAND|wx.CENTER, 2)
		whole_box.Add(test_box_Sizer,0, wx.EXPAND|wx.CENTER, 2)
		p.SetSizer(whole_box)
	#def InitTensorflow(self):
		
	"""
		x=tf.placeholder(dtype=tf.float32,shape=[None,28,28])
		y=tf.placeholder(dtype=tf.float32,shape=[None,n_classes])

		weights=tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
		biases=tf.Variable(tf.random_normal([n_classes]))
		pred=BIRNN(x,weights,biases)

		pred_number=tf.argmax(pred,1)

		lost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

		optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lost)

		correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

		accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
		"""

	def OnClick(self,event):
		#self.tx.SetLabel('wudi\r\nsb')
		#print('click button')
		tg_count=0
		init_string='--------strat train--------\r\n'
		self.tx.AppendText(init_string)
		self.tg.SetValue(tg_count)
		
		#define tf.session
		sess=tf.Session()
		sess.run(init)
		step=1
		#tf.summary.FileWriter("logs/",sess.graph)
		train_data=mnist.train.images
		train_label=mnist.train.labels
		train_acc=[]
		eval_indices=[]
		while step*batch_size<max_samples:
			#batch_x,batch_y=mnist.train.next_batch(batch_size)
			#batch_x=batch_x.reshape((batch_size,n_steps,n_input))
			rand_index = np.random.choice(len(train_data), size=batch_size)
			batch_x=train_data[rand_index].reshape((batch_size,n_steps,n_input))
			batch_y=train_label[rand_index]
			batch_y=generate_one_hot(batch_y,batch_size,n_classes) #one_hot encoding
			#print('batch_x/shape:'+str(batch_x.shape))
			sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
			if step%display_step==0:
				acc=sess.run(accuracy,{x:batch_x,y:batch_y})
				loss=sess.run(lost,{x:batch_x,y:batch_y})
				eval_indices.append(step)
				train_acc.append(loss)
				
				second_string=('step:'+str(step*batch_size)+' loss='+str(loss)+' acc in train'+str(acc)+'\r\n')
				print(second_string)
				
				tg_count=tg_count+1
				self.tx.AppendText(second_string)#设置文本
				self.tg.SetValue(tg_count)
				
				time.sleep(0.8);
			
			step+=1
			#训练结束,修改标记和画出图像
		self.trainTrigger=True
		
		
		plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
		plt.title('Train Accuracy')
		plt.xlabel('step of Generation')
		plt.ylabel('Accuracy')
		plt.legend(loc='upper right')
		savefig('curve.jpg')
		#图片绘画结束
		curve_Image=wx.Image("curve.jpg",wx.BITMAP_TYPE_JPEG).Scale(200,300).ConvertToBitmap()
		self.bmp.SetBitmap(curve_Image)
		
	def Oncombo(self,event):
		if(self.trainTrigger==True):
			#matplotlib.use('Agg')
		
			Have_choice=int(self.combo.GetValue())
			Hc_index=random_down_index(Have_choice)#随机挑一个
		
			strings=('pred: %d'%Have_choice) #设置标签
			self.st.SetLabel(strings)
		
			Hc_show_data=mnist.test.images[Hc_index].reshape(28,28) #画图用数据
			plt.imshow(Hc_show_data)
			plt.title('')
			plt.xlabel('')
			plt.ylabel('')
			savefig('num.jpg')
			number_Image=wx.Image("num.jpg",wx.BITMAP_TYPE_JPEG).Scale(200,300).ConvertToBitmap()
			self.cbmp.SetBitmap(number_Image)
	

app = wx.App() 
Example(None, title = 'MNIST Recongizer') 
app.MainLoop()