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
LOAD_GRAPH_NAME = "three_moves_400_epochs_3.tfg"
SAVE_GRAPH_NAME = "three_moves_400_epochs_4.tfg"
TRAINING = False
PASS_PER_DATASET = 1
EPOCHS = 400

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
	Imitation Learning
'''

class ImitationLearner():

	def __init__(self):

		self.initialize_vars = False
		self.EPOCHS = PASS_PER_DATASET
		self.train_losses = []
		self.valid_losses = []

		self.in_frame = tf.placeholder(tf.float32, (None, 100,100,3), name='in_frame')
		# expecting a one-hot vector
		self.in_action = tf.placeholder(tf.uint8, (None, 3), name='in_action')

		# ask thomas about whitening

		h = self.in_frame
		for i, n in enumerate([10,20,30,40,50,60]):
		    h = tf.contrib.layers.conv2d(h, n, (3,3), stride=2, scope="conv%d"%i, weights_regularizer=tf.nn.l2_loss)
		    print(h)
		h = tf.contrib.layers.flatten(h)
		print(h)
		self.action_logit = tf.identity(tf.contrib.layers.fully_connected(h, 3, activation_fn=None), name='action_logit')
		self.action_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.action_logit, labels=tf.cast(self.in_action, tf.float32))
		self.total_loss = self.action_loss + 1e-6 * tf.losses.get_regularization_loss()
		optimizer = tf.train.AdamOptimizer(.001, .9, .999)
		self.opt = optimizer.minimize(self.total_loss)
 
	def train(self, training_data, validation_data):
		if not self.initialize_vars:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			self.initialize_vars = True
		actions, states, frames = training_data
		# print(type(frames), frames.shape)
		for e in range(self.EPOCHS):
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
		util.save(SAVE_GRAPH_NAME,session=self.sess)
		self.train_losses = []
		self.valid_losses = []
		# return r

def test_predict_action(sess,frame):
	#tf.get_default_graph().get_tensor_by_name("in_frame:0")
	graph = tf.get_default_graph()
	PA = sess.run([graph.get_tensor_by_name("action_logit:0")], {graph.get_tensor_by_name("in_frame:0"):frame})[0][0]
	# print(PA)
	action = ''
	for b in PA:
		action+=str(int(b > 0))
	r = int(action, 2)
	# print ("action: ", r)
	return r


if TRAINING:
	il = ImitationLearner()
	data = np.array([])
	#valid_data = np.load("training7.npy")
	for x in range(EPOCHS):
		print ("EPOCH: %d" % x)
		for i in range(4,8):
			print("training on data %d"%i)
			d = np.load("training%d.npy"%i)
			il.train(parser(d),parser(d))
	#for i in range(t_12.shape[0]):
	#	np.insert(t_11,t_11.shape[0],t_12[i])
	# il.train(parser(t_11),parser(t_11))

else:
	sess = tf.Session()
	util.load(LOAD_GRAPH_NAME, session=sess)

# play the game
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
file = open('actions.txt', 'w+')
for i in range(10000):
	speed = state['speed']
	wrongway = state['wrongway']
	if speed == 0 or wrongway > 0:
		if stuck_count == 20:
			# Then, each time you generate an action
			file.write(str(64) + ':' + str(state['timestamp']) + '\n')
			state, obs = K.step(64)
			stuck_count = 0
		else:
			stuck_count+=1

	n = 128 if np.random.rand() > .8 else 0
	action = n + test_predict_action(sess, [skimage.transform.resize(obs, (100, 100))])
	# Then, each time you generate an action
	file.write(str(action) + ':' + str(state['timestamp']) + '\n')
	state, obs = K.step(action)
	print( action )
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
