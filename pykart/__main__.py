from pykart import Kart
from time import sleep, time
from random import random

t0 = time()
K = Kart("lighthouse", 300, 200)
t1 = time()
K.restart()
t2 = time()
print( K.waitRunning() )
t3 = time()
print( K.running, t1-t0, t2-t1, t3-t2 )

im, lbl = None, None
fwd, left, right = 1, 0, 0
# implement Neural Net here
"""
			state["position_along_track"] = sh->position_along_track;
			state["distance_to_center"] = sh->distance_to_center;
			state["position_in_race"] = sh->position_in_race;
			state["speed"] = sh->speed;
			state["smooth_speed"] = sh->smooth_speed;
			state["wrongway"] = sh->wrongway;
			state["energy"] = sh->energy;
			state["finish_time"] = sh->finish_time;
			state["angle"] = sh->angle;
			state["timestamp"] = sh->timestamp;

"""
for i in range(10000):
	state, obs = K.step(4*fwd + 1 * left + 2 * right)
	# obs is 300x200x3 image
	# TODO: feed obs into conv net and extract features
	left = -state['angle'] + 0.1*state['distance_to_center'] > .1# or state['distance_to_center'] > 0.2
	right = -state['angle'] + 0.1*state['distance_to_center'] < -.1# or state['distance_to_center'] < -0.2
	fwd = 1
	if abs(state['angle']) > 0.4:
		fwd = random()>0.5
	
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
