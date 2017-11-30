from pykart import Kart
from time import sleep, time
from random import random
import tensorflow as tf


def play_level(policy, K):
	t0 = time()
	t1 = time()
	K.restart()
	t2 = time()
	print( K.waitRunning() )
	t3 = time()
	print( K.running, t1-t0, t2-t1, t3-t2 )

	im, lbl = None, None
	fwd, left, right = 1, 0, 0

	pos_along_track = 0
	action = 4*fwd + 1*left + 2*right
	for i in range(10000):
		state, obs = K.step(action)
		action = policy(state, obs)
		# obs is 300x200x3 image
		# TODO: feed obs into conv net and extract features
		# left = -state['angle'] + 0.1*state['distance_to_center'] > .1# or state['distance_to_center'] > 0.2
		# right = -state['angle'] + 0.1*state['distance_to_center'] < -.1# or state['distance_to_center'] < -0.2
		# fwd = 1
		# if abs(state['angle']) > 0.4:
		# 	fwd = random()>0.5
		
		print( state )
		from pylab import *
		ion()
		# figure()
		#subplot(1,2,1)
		if obs is not None:
			if im is None:
				im = imshow(obs)
			else:
				im.set_data(obs)
		draw()
		pause(0.001)

def score(state):
	# position along track,
	# distance to center, position_in_race, 
	# speed, smooth_speed, wrongway, 
	# energy, finish_time, angle, timestamp
	score = state['position_along_track'] * 10 
			+ state['speed'] * 10
			- state['wrongway'] * 50
	return score
# implement Neural Net here

# Whiten the image
# TODO: Ask Thomas about whitening image
# white_image = (image - 100.) / 72.
class NN:
	def __init__(self):
		self.I = tf.placeholder(tf.uint8, (None,300,200,3), name='input')
		self.state = tf.placeholder(tf.float32, (None,8), name='state')
		h = tf.cast(self.I, tf.float32)
		C0 = 8
		D = 6
		hs = []
		# print(h)
		# Encode image
		for i in range(D):
			hs.append(h)
			h = tf.contrib.layers.conv2d(h, 
				C0*int(1.5**i), 
				(3,3), 
				stride=2, 
				scope="ae_conv2d%d"%(i+1), 
				weights_regularizer=tf.nn.l2_loss)
			# print(h)
		# Decode image
		for i in range(D)[::-1]:
			h = tf.contrib.layers.conv2d_transpose(h, 
				C0*int(1.5**i), 
				(3,3), 
				stride=2, 
				scope="ae_upconv2d%d"%(i+1), 
				weights_regularizer=tf.nn.l2_loss)
			# h = tf.concat([h, hs[i]], axis=-1) # skip connection
			# print(h)

		flattened = tf.contrib.layers.flatten (h)

		# combine with states here
		flattened = flattened * self.state 
		self.action_logit = tf.contrib.layers.fully_connected(
			inputs=flattened,
			num_outputs=8,
			weights_regularizer=tf.nn.l2_loss,
			activation_fn=None,
			scope="conv_output") # This should be a (None,8) tensor

	def __call__(self, state, I):
		#TODO: decide what paramters here
		pred_action = sess.run(self.action_logit, {self.I: I[None]})
		action = [i > .5 for i in pred_action]
		sum = 0
		for a in range(len(action)):
			sum += action[a] * (2**a)
		return sum


K = Kart("lighthouse", 300, 200)
CNN = NN()
sess.run(tf.global_variables_initializer())
for i in range(1):
	play_level(CNN, K)

