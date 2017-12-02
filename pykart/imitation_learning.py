from pykart import Kart
from time import sleep, time
from random import random
import tensorflow as tf
import numpy as np
import skimage.transform


VALID_ACTIONS = 8
STATES = 10

def parser(file):
	data = np.load(file)
	actions, states, frames = [], [], []
	for d in data:
		a, s, f = d
		
		# convert the action to a one-hot of booleans
		bitsring = "{0:b}".format(a)
		act = []
		for b in bitsring:
			act.append(int(b)>0)
		while len(act) < 8:
			act.insert(0,False)
		actions.append(act)

		# convert the states into a list
		states.append(list(s.values()))
		frames.append(f)
	# todo: parse action into a one-hot and convert states dict to an array
	return actions, states, frames

class ImitationLearner():

	def __init__(self):

		self.initialize_vars = False
		self.EPOCHS = 50
		self.train_losses = []
		self.valid_losses = []


		self.in_frame = tf.placeholder(tf.float32, (None, 100,100,3), name='in_frame')
		self.in_state = tf.placeholder(tf.float32, (None, STATES), name='in_state')
		# expecting a one-hot vector
		self.in_action = tf.placeholder(tf.uint8, (None, VALID_ACTIONS), name='in_action')

		# ask thomas about whitening

		h = self.in_frame
		for i, n in enumerate([10,20,30,40,50,60]):
		    h = tf.contrib.layers.conv2d(h, n, (3,3), stride=2, scope="conv%d"%i, weights_regularizer=tf.nn.l2_loss)
		    print(h)
		h = tf.contrib.layers.flatten(h)
		print(h.shape, self.in_state.shape)
		# h = tf.transpose(h) * self.in_state
		print(h)
		self.action_logit = tf.contrib.layers.fully_connected(h, VALID_ACTIONS, activation_fn=None)
		self.state_logit = tf.contrib.layers.fully_connected(h, STATES, activation_fn=None)
		self.action_loss = tf.losses.sigmoid_cross_entropy(logits=self.action_logit, multi_class_labels=self.in_action)
		self.state_loss = tf.losses.mean_squared_error(labels=self.in_state, predictions=self.state_logit)
		self.total_loss = self.action_loss + tf.losses.get_regularization_loss()
		optimizer = tf.train.AdamOptimizer(.001, .9, .999)
		self.opt = optimizer.minimize(self.total_loss)
 
	def train(self, training_data, validation_data):
		if not self.initialize_vars:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			self.initialize_vars = True

		for e in range(self.EPOCHS):

			# do some training
			for i in range(10):
				train_loss, _ = self.sess.run([self.total_loss, self.opt], 
					{self.in_action:training_data[0],
					 self.in_state:training_data[1],
					 self.in_frame:training_data[2]})
				self.train_losses.append(train_loss)
			print("training loss: ", np.mean(self.train_losses))
			# do some validation
			validation_loss = self.sess.run([self.total_loss],
					{self.in_action:validation_data[0],
					 self.in_state:validation_data[1],
					 self.in_frame:validation_data[2]})
			self.valid_losses.append(validation_loss)
			print("validation loss: ", np.mean(self.valid_losses))

	def predict_action(self, frame):
		pred_action = self.sess.run([self.action_logit], {self.in_frame: frame})
		bool_act = ''
		for a in pred_action[0][0]:
			bool_act += str(int(a > 0))
		r = int(bool_act, 2)
		print("********************ACTION**************: ",r)
		return r



il = ImitationLearner()
train_d = parser("training.npz.npy")
valid_d = parser("training.npz.npy")
il.train(train_d,valid_d)

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

for i in range(10000):
	
	state, obs = K.step(il.predict_action([skimage.transform.resize(obs, (100, 100))]))
	
	print( state )
	from pylab import *
	ion()
	#figure()
	#subplot(1,2,1)
	if obs is not None:
		if im is None:
			im = imshow(obs)
		else:
			im.set_data(obs)
	draw()
	pause(0.001)
