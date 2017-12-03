from pykart import Kart
from time import sleep, time
from random import random
import tensorflow as tf
import numpy as np
import skimage.transform
from pylab import *
import util


VALID_ACTIONS = 8
STATES = 10
GRAPH_NAME = "three-moves.tfg"
TRAINING = False

def parser(data):
	# data = np.load(file)
	actions, states, frames = [], [], []
	four = five = six = 0
	for d in data:
		a, s, f = d
		
		# convert the action to a one-hot of booleans
		bitsring = "{0:b}".format(a)
		act = []
		for b in bitsring:
			act.append(int(b)>0)
		while len(act) < 8:
			act.insert(0,False)

		if a == 4:
			four+=1
			act = [True, False, False]
		elif a == 5:
			five+=1
			act = [True, False, True ]
		elif a == 6:
			six+=1
			act = [True, True, True]
		actions.append(act)
		# convert the states into a list
		states.append(list(s.values()))
		frames.append(f)
	return actions, states, np.array(frames)

'''
	Gradient Free RL 
'''

class GradientFreeRL():
	def __init__(self):
		self.sess = tf.Session()
		self.initialize_vars = True

		self.I = tf.placeholder(tf.float32, (None,100,100,3), name='input')
		
		# TODO: Define your convnet
		# You don't need an auxiliary auto-encoder loss, just create
		# a few encoding conv layers.
		outs = [8,12,16,20,24,8]
		conv = self.I
		for i in range(6):
			conv = tf.contrib.layers.conv2d(conv, outs[i], (3,3), stride=2, weights_regularizer=tf.nn.l2_loss, scope="act_conv2d{}".format(i+1))
		self.action_logit = tf.contrib.layers.fully_connected(tf.layers.flatten(conv), 3, activation_fn=None)
		print(self.action_logit)

		
		self.predicted_action = tf.identity((self.action_logit), name='action')
		self.variables =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		self.__vars_ph = [tf.placeholder(tf.float32, v.get_shape()) for v in self.variables]
		self.__assign = [v.assign(p) for v,p in zip(self.variables, self.__vars_ph)]
		self.sess.run(tf.global_variables_initializer())
	
	def __call__(self, I, greedy_eps=1e-5, verbose=False):
		PA = list(self.sess.run(self.predicted_action, {self.I: I})[0])
		action = max(PA)

		if PA.index(action) == 0:
			# print("forward")
			return 4 	
		elif PA.index(action) == 1:
			# print("left")
			return 5
		elif PA.index(action) == 2:
			# print("right")
			return 6
	
	def sess(self):
		return self.sess

	@property
	def flat_weights(self):
		import numpy as np
		W = self.weights
		return np.concatenate([w.flat for w in W])
	
	@flat_weights.setter
	def flat_weights(self, w):
		import numpy as np
		S = [v.get_shape().as_list() for v in self.variables]
		s = [np.prod(i) for i in S]
		O = [0] + list(np.cumsum(s))
		assert O[-1] <= w.size
		W = [w[O[i]:O[i+1]].reshape(S[i]) for i in range(len(S))]
		self.weights = W
	
	@property
	def weights(self):
		return self.sess.run(self.variables)
	
	@weights.setter
	def weights(self, weights):
		self.sess.run(self.__assign, {v:w for v,w in zip(self.__vars_ph, weights)})

def score_state(state):
	# position along track,
	# distance to center, position_in_race, 
	# speed, smooth_speed, wrongway, 
	# energy, finish_time, angle, timestamp
	score = state['position_along_track'] #- abs(state['distance_to_center']) * 10
	return score

def score_policy(agent):

	#start the game
	K = Kart("lighthouse", 500, 500)
	K.restart()
	print( K.waitRunning() )
	sleep(1)
	state, obs = K.step(4)

	# keep track of each state's score
	scores = []
	# play the game a bunch
	speed = state['speed']
	max_pos = position = state['position_along_track']
	for i in range(10000):
		if (i + 1)%100 == 0 and speed == 0:
			break

		state, obs = K.step(agent([skimage.transform.resize(obs, (100, 100))]))
		speed = state['speed']
		position = state['position_along_track']
		if position > max_pos:
			position = max_pos
		else:
			break
		scores.append(score_state(state))
		# print( state )
		# ion()
		#figure()
		#subplot(1,2,1)
		# if obs is not None:
		# 	if im is None:
		# 		im = imshow(obs)
		# 	else:
		# 		im.set_data(obs)
		# draw()
		sleep(0.001)
	return scores[-1], max(scores)

# gfrl_agent = GradientFreeRL()

# N = SAMPLES_PER_EPOCH = 50
# EPOCHS = 5
# SURVIVAL_RATE = 0.8
# VARIANCE_EPS = 0.1
# EVOLUTION_RATE = .0001
# NUM_VARS = gfrl_agent.flat_weights.size
# mean = np.zeros(gfrl_agent.flat_weights.shape)
# std = 0*mean + 1
# agents = np.array([np.random.normal(loc=mean,scale=std) for i in range(N)])

# # f(x) evaluates how well a certain parameter setting works
# def f(agent, x):
# 	agent.flat_weights = x
# 	return np.sum(score_policy(agent))

# max_score = .3
# best_agent = agents[0]

# for epoch in range(5):
# 	print("epoch: {}".format(epoch))
# 	for a in range(len(agents)):
# 		score = f(gfrl_agent, agents[a])
# 		if score > max_score:
# 			best_agent = agents[a]
# 			max_score = score
# 			print("max_score: {}".format(max_score))
# 			gfrl_agent.flat_weights = best_agent
# 			util.save('GradientFreeRL-{}.tfg'.format(max_score), session=gfrl_agent.sess)
# 		else:
# 			agents[a] = np.random.normal(loc=mean,scale=std)


'''
	Imitation Learning
'''

class ImitationLearner():

	def __init__(self):

		self.initialize_vars = False
		self.EPOCHS = 100
		self.train_losses = []
		self.valid_losses = []


		self.in_frame = tf.placeholder(tf.float32, (None, 100,100,3), name='in_frame')
		self.in_state = tf.placeholder(tf.float32, (None, STATES), name='in_state')
		# expecting a one-hot vector
		self.in_action = tf.placeholder(tf.uint8, (None, 3), name='in_action')

		# ask thomas about whitening

		h = self.in_frame
		for i, n in enumerate([10,20,30,40,50,60]):
		    h = tf.contrib.layers.conv2d(h, n, (3,3), stride=2, scope="conv%d"%i, weights_regularizer=tf.nn.l2_loss)
		    print(h)
		h = tf.contrib.layers.flatten(h)
		print(h.shape, self.in_state.shape)
		# h = tf.transpose(h) * self.in_state
		print(h)
		self.action_logit = tf.identity(tf.contrib.layers.fully_connected(h, 3, activation_fn=None), name='action_logit')
		self.state_logit = tf.contrib.layers.fully_connected(h, STATES, activation_fn=None)
		self.action_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.action_logit, labels=tf.cast(self.in_action, tf.float32))
		self.state_loss = tf.losses.mean_squared_error(labels=self.in_state, predictions=self.state_logit)
		self.total_loss = self.action_loss + 1e-6 * tf.losses.get_regularization_loss()
		optimizer = tf.train.AdamOptimizer(.001, .9, .999)
		self.opt = optimizer.minimize(self.total_loss)
 
	def train(self, training_data, validation_data):
		if not self.initialize_vars:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			self.initialize_vars = True
		actions, states, frames = training_data
		print(type(frames), frames.shape)
		for e in range(self.EPOCHS):
			print("Epoch: ", e)
			# do some training
			for i in range(10):
				train_loss, _ = self.sess.run([self.total_loss, self.opt], 
					{self.in_action:actions,
					 self.in_state:states,
					 self.in_frame:frames})
				self.train_losses.append(train_loss)
			print("training loss: ", np.mean(self.train_losses))
			# do some validation
			validation_loss = self.sess.run([self.total_loss],
					{self.in_action:validation_data[0],
					 self.in_state:validation_data[1],
					 self.in_frame:validation_data[2]})
			self.valid_losses.append(validation_loss)
			print("validation loss: ", np.mean(self.valid_losses))
		util.save("three-moves_100_epochs.tfg",session=self.sess)

	def predict_action(self, frame):
		PA = self.sess.run([self.action_logit], {self.in_frame: frame})[0][0]
		print(PA)
		action = ''
		for b in PA:
			action+=str(int(b > 0))
		r = int(action, 2)
		return r
		# bool_act = ''
		# for a in pred_action[0][0]:
		# 	bool_act += str(int(a > 0))
		# r = int(bool_act, 2)
		# print("********************ACTION**************: ",r)
		# return r

def test_predict_action(sess,frame):
	#tf.get_default_graph().get_tensor_by_name("in_frame:0")
	graph = tf.get_default_graph()
	PA = sess.run([graph.get_tensor_by_name("action_logit:0")], {graph.get_tensor_by_name("in_frame:0"):frame})[0][0]
	print(PA)
	action = ''
	for b in PA:
		action+=str(int(b > 0))
	r = int(action, 2)
	print ("action: ", r)
	return r


if TRAINING:
	il = ImitationLearner()
	# train_d = parser("training11.npy")
	# train_1 = parser("kart-data/training1.npy")
	# train_2 = parser("kart-data/training2.npy")
	# train_3 = parser("kart-data/training3.npy")
	# valid_d = parser("training.npy")
	# il.train(train_d, valid_d)
	# il.train(train_1, valid_d)
	# il.train(train_2, valid_d)
	# il.train(train_3, valid_d)
	# 	il.train(d,parser("training.npy"))
		
	t_11 = np.load("racing_data.npy")
	#t_12 = np.load("training12.npy")
	# t_1 = np.load("kart-data/training1.npy")
	# t_2 = np.load("kart-data/training2.npy")
	# t_3 = np.load("kart-data/training3.npy")
	#for i in range(t_12.shape[0]):
	#	np.insert(t_11,t_11.shape[0],t_12[i])
	# t_14 = np.append(t_12, t_2, axis = 0)
	# t_all = np.append(t_12, t_3, axis = 0)
	il.train(parser(t_11),parser(t_11))

	t0 = time()
	K = Kart("lighthouse", 500, 500)
	t1 = time()
	K.restart()
	t2 = time()
	print( K.waitRunning() )
	t3 = time()
	print( K.running, t1-t0, t2-t1, t3-t2 )
	sleep(1)

	im, lbl = None, None
	fwd, left, right = 1, 0, 0
	# 4 = go forward
	state, obs = K.step(4)
	stuck_count = 0
	for i in range(10000):
		speed = state['speed']
		if speed == 0:
			if stuck_count == 10:
				state, obs = K.step(64)
				stuck_count = 0
			else:
				stuck_count+=1

		state, obs = K.step(il.predict_action([skimage.transform.resize(obs, (100, 100))]))
		# print( state )
		from pylab import *
		ion()
		# figure()
		# subplot(1,2,1)
		if obs is not None:
			if im is None:
				im = imshow(obs)
			else:
				im.set_data(obs)
		draw()
		pause(0.001)
else:
	sess = tf.Session()
	util.load(GRAPH_NAME, session=sess)
	t0 = time()
	K = Kart("lighthouse", 500, 500)
	t1 = time()
	K.restart()
	t2 = time()
	print( K.waitRunning() )
	t3 = time()
	print( K.running, t1-t0, t2-t1, t3-t2 )
	sleep(1)

	im, lbl = None, None
	fwd, left, right = 1, 0, 0
	# 4 = go forward
	state, obs = K.step(4)
	stuck_count = 0
	for i in range(10000):
		speed = state['speed']
		if speed == 0:
			if stuck_count == 10:
				state, obs = K.step(64)
				stuck_count = 0
			else:
				stuck_count+=1

		state, obs = K.step(test_predict_action(sess, [skimage.transform.resize(obs, (100, 100))]))
		# print( state )
		from pylab import *
		ion()
		# figure()
		# subplot(1,2,1)
		if obs is not None:
			if im is None:
				im = imshow(obs)
			else:
				im.set_data(obs)
		draw()
		pause(0.001)
