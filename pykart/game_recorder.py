#!/usr/bin/python3
from pykart import Kart
from time import sleep, time
from random import random
import _thread
import readchar
from pylab import *
import queue
import skimage.transform	

global q
q = queue.Queue()

def key_logger(unused):
	global q
	global go
	go = True
	left = 0
	right = 0
	fwd = 0
	while go:
		key = str(readchar.readchar())
		#print (repr(readchar.readchar()))
		if key == 'a': # this is left
			if (right == 1):
				left = 0
				right = 0
			else:
				left = 1
				right = 0
		elif key == 'd': # this is right
			if (left == 1):
				left = 0
				right = 0
			else:
				left = 0
				right = 1
		elif key == 'w':
			fwd = 1
			back = 0
		elif key == 's':
			back = 1
			fwd = 0
		elif key == '\x03':
			# quit
			go = False
			print ("quitting")
		action = 4*fwd + 1*left + 2*right + back*8
		q.put(action)

def play_game(unused):
	#t0 = time()
	K = Kart("lighthouse", 500, 500)
	#t1 = time()
	K.restart()
	#t2 = time()
	print( K.waitRunning() )
	sleep(1)
	#t3 = time()
	#print( K.running, t1-t0, t2-t1, t3-t2 )
	global q
	global go
	im, lbl = None, None
	action = 0
	data = []
	while go:
		while (q.empty()):
			sleep(0.01)
		if not q.empty():
			action = q.get()
			print ("action: ", action)
		# else:
		# 	sleep (1)
		# 	action = 0
		# 	print ("action: ", action)
		state, obs = K.step(action)
		frame = skimage.transform.resize(obs, (100, 100))
		data.append([action,state,frame])
		# print (state)
		ion()
		#figure()
		subplot(1,2,1)
		if obs is not None:
			if im is None:
				im = imshow(obs)
			else:
				im.set_data(obs)
		draw()
		sleep(0.01) # one millisecond?
	K.quit()
	np.save("training.npz",data)


try:
	_thread.start_new_thread( key_logger, (0,))
	_thread.start_new_thread( play_game, (0,))
except:
	print ("Error: unable to start thread")

while 1:
	pass