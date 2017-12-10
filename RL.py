import numpy as np
import tensorflow as tf
import gym 
import time
env=gym.make('CartPole-v0')
env.reset()
random_episodes=0
reward_sum=0
while random_episodes<10:
	env.render()
	observe,reward,done,_=env.step(np.random.randint(0,2))
	reward_sum+=reward
	if done:
		random_episodes+=1
		print('reward for this episode was:',reward_sum)
		
		reward_sum=0
		time.sleep(0.5)
		#print('observation'+str(observation))
		env.reset()

H=5
batch_size=25
learning_rate=0.1
D=4
gamma=0.99

#create a policy network using MLP
input_x=tf.placeholder(dtype=tf.float32,shape=[None,D])
W1=tf.Variable(tf.random_normal(shape=[D,H],dtype=tf.float32))
layer1=tf.nn.relu(tf.matmul(input_x,W1))
W2=tf.Variable(tf.random_normal(shape=[H,1],dtype=tf.float32))
score=tf.matmul(layer1,W2)
probability=tf.nn.sigmoid(score)

#batch_size training compute loss
input_y=tf.placeholder(dtype=tf.float32,shape=[None,1])
advantages=tf.placeholder(dtype=tf.float32,shape=[None,1])
loglik=tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability))
loss=-tf.reduce_mean(loglik*advantages)

#tvars=tf.trainable_variables()
#optimizer
adam=tf.train.AdamOptimizer(learning_rate)
train=adam.minimize(loss)

def discount_rewards(r):
	discounted_r=np.zeros_like(r)
	running_add=0
	for t in reversed(range(r.size)):
		running_add=running_add*gamma+r[t]
		discounted_r[t]=running_add
	return discounted_r



#training
xs,ys,drs=[],[],[]
reward_sum=0
episode_number=1
total_episode=10000

sess=tf.Session()
rendering=False
init=tf.global_variables_initializer()
sess.run(init)
observation=env.reset() #observation to environment
while episode_number<=total_episode:
	if reward_sum>100 or rendering==True:
		env.render()
		rendering=True
	x=np.reshape(observation,[1,D])
	tfprob=sess.run(probability,{input_x:x})
	action=1 if np.random.uniform()<tfprob else 0
	xs.append(x)
	y=1-action
	ys.append(y)
	observation,reward,done,info=env.step(action)
	reward_sum+=reward
	drs.append(reward)
	
	if done:  #game over & do a epoch traning
		
		epx=np.vstack(xs)                                            
		epy=np.vstack(ys)
		epr=np.vstack(drs)
		xs,ys,drs=[],[],[]
		
		discounted_epr=discount_rewards(epr)
		discounted_epr-=np.mean(discounted_epr)
		discounted_epr/=np.std(discounted_epr)
		
		sess.run(train,{input_x:epx,input_y:epy,advantages:discounted_epr})
		print('average reward in %d epoch is %f'%(episode_number,reward_sum))
		
		episode_number+=1
		reward_sum=0
		observation=env.reset()