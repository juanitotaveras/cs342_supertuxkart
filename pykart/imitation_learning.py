from pykart import Kart
from time import sleep, time
from random import random
import tensorflow as tf
import numpy as np
import skimage.transform
import util

TEST = False

VALID_ACTIONS = 8
STATES = 10

TRACK_POS = 0
DIST_TO_CENTER = 1
SPEED = 3
ANGLE = 8

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
		while len(act) < VALID_ACTIONS:
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
		self.EPOCHS = 10
		self.train_losses = []
		self.state_losses = []
		self.valid_losses = []

		self.in_frame = tf.placeholder(tf.float32, (None, 100,100,3), name='in_frame')
		self.in_speed = tf.placeholder(tf.float32, (None, 1), name='in_speed')
		self.in_dist_center = tf.placeholder(tf.float32, (None, 1), name='in_dist_center')
		self.in_pos = tf.placeholder(tf.float32, (None, 1), name='in_pos')
		self.in_angle = tf.placeholder(tf.float32, (None, 1), name='in_angle')
		# expecting a one-hot vector
		self.in_action = tf.placeholder(tf.uint8, (None, VALID_ACTIONS), name='in_action')

		# ask thomas about whitening

		h = self.in_frame
		for i, n in enumerate([10,20,30,40,50,60]):
		    h = tf.contrib.layers.conv2d(h, n, (3,3), stride=2, scope="conv_%d"%i, weights_regularizer=tf.nn.l2_loss)
		    print(h)
		h = tf.contrib.layers.flatten(h)

		# h = tf.transpose(h) * self.in_state
		print(h)
		shared_fc = tf.contrib.layers.fully_connected(h, 10, activation_fn=None)
		# shared_fc = h
		speed_logit = tf.contrib.layers.fully_connected(shared_fc, 1, activation_fn=None)
		speed_loss = tf.reduce_mean(tf.nn.l2_loss(speed_logit - self.in_speed))
		dist_logit = tf.contrib.layers.fully_connected(h, 10)
		dist_logit = tf.contrib.layers.fully_connected(dist_logit, 1, activation_fn=None)
		dist_loss = tf.losses.mean_squared_error(predictions=dist_logit, labels=self.in_dist_center)
		pos_logit = tf.contrib.layers.fully_connected(h, 10)
		pos_logit = tf.contrib.layers.fully_connected(pos_logit, 1, activation_fn=None)
		pos_loss = tf.losses.mean_squared_error(labels=self.in_pos, predictions=pos_logit)
		angle_logit = tf.contrib.layers.fully_connected(shared_fc, 1, activation_fn=None)
		angle_loss = tf.reduce_mean(tf.nn.l2_loss(angle_logit - self.in_angle))
		self.action_logit = tf.contrib.layers.fully_connected(h, VALID_ACTIONS, activation_fn=None)
		self.action_logit = tf.identity(self.action_logit, name='action_logit')
		self.action_loss = tf.losses.sigmoid_cross_entropy(logits=self.action_logit, multi_class_labels=self.in_action)
		# self.state_logit = tf.contrib.layers.fully_connected(h, STATES, activation_fn=None)
		# self.state_loss = tf.losses.mean_squared_error(labels=self.in_state, predictions=self.state_logit)
		# self.state_loss = 1e-3 * speed_loss + 10 * dist_loss + pos_loss + 1e-3 * angle_loss
		self.state_loss = pos_loss
		self.total_loss = self.action_loss + 1e-3*self.state_loss #+ 1e-6*tf.losses.get_regularization_loss()
		optimizer = tf.train.AdamOptimizer(.001, .9, .999)
		self.opt = optimizer.minimize(self.total_loss)
		print( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]) )
 
	def format_state_info(self, training_data):
		track_poss = np.array([training_data[1][i][TRACK_POS] for i in range(len(training_data[1]))])
		track_poss.shape = track_poss.shape[0], 1 
		dists = np.array([training_data[1][i][DIST_TO_CENTER] for i in range(len(training_data[1]))])
		dists.shape = dists.shape[0], 1
		speeds = np.array([training_data[1][i][SPEED] for i in range(len(training_data[1]))])
		speeds.shape = speeds.shape[0], 1
		angles = np.array([training_data[1][i][ANGLE] for i in range(len(training_data[1]))])
		angles.shape = angles.shape[0],1
		return track_poss, dists, speeds, angles

	def train(self, training_data, validation_data):
		if not self.initialize_vars:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			self.initialize_vars = True
		for e in range(self.EPOCHS):
			print("EPOCH: ",e)
			# do some training
			t_pos, t_dist, t_speed, t_angle = self.format_state_info(training_data)
			v_pos, v_dist, v_speed, v_angle = self.format_state_info(validation_data) 
			for i in range(10):
				train_loss, state_loss, _ = self.sess.run([self.total_loss, self.state_loss, self.opt], 
					{self.in_action:training_data[0],
					 self.in_frame:training_data[2],
					 self.in_speed:t_speed,
					 self.in_pos:t_pos,
					 self.in_dist_center:t_dist,
					 self.in_angle:t_angle})
				self.state_losses.append(state_loss)
				self.train_losses.append(train_loss)
			print("training loss: ", np.mean(self.train_losses))
			print("state loss: ", np.mean(self.state_losses))
			# do some validation
			validation_loss = self.sess.run([self.total_loss],
					{self.in_action:training_data[0],
					 self.in_frame:training_data[2],
					 self.in_speed:v_speed,
					 self.in_pos:v_pos,
					 self.in_dist_center:v_dist,
					 self.in_angle:v_angle})
			self.valid_losses.append(validation_loss)
			print("validation loss: ", np.mean(self.valid_losses))
		util.save("supertuxkart-graph2.tfg", session=self.sess)

	def predict_action(self, frame):
		pred_action = self.sess.run([self.action_logit], {self.in_frame: frame})
		bool_act = ''
		for a in pred_action[0][0]:
			bool_act += str(int(a > 0))
		r = int(bool_act, 2)
		print("********************ACTION**************: ",r)
		return r

def test_predict_action(sess, action_logit, in_frame, frame):
	p_action = sess.run([action_logit], {in_frame: frame})
	bool_act = ''
	print (p_action)
	for a in p_action[0][0]:
		bool_act += str(int(a > 0))
	r = int(bool_act, 2)
	print("********************ACTION**************: ",r)
	return r

if TEST:
	tf.reset_default_graph()
	#saver = tf.train.Saver()

	#saver.restore('supertuxkart-graph.tfg')
	sess = tf.Session()
	util.load ('supertuxkart-graph2.tfg', session=sess) 
	#graph = tf.get_default_graph()

	action_logit = graph.get_tensor_by_name('action_logit:0')
	in_frame = graph.get_tensor_by_name('in_frame:0')
	sess.run(tf.global_variables_initializer())
	for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		print (i)
	print( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]) )
	print(tf.trainable_variables())

else:
	il = ImitationLearner()
	train_d = parser("training.npy")
	valid_d = parser("training.npy")
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
	if TEST:
		state, obs = K.step(test_predict_action(sess, action_logit, in_frame, [skimage.transform.resize(obs, (100, 100))]))
	else:
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
