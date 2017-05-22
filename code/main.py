import os
import sys
import argparse
import tensorflow as tf
from model import *

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode',dest='forward_only',help='Test or Train Mode',default=0,type=int)
	parser.add_argument('-b','--batch_size',dest='batch_size',help='Batch Size for Training',default=32,type=int)
	parser.add_argument('-e','--num_epochs',dest='num_epochs',help='Number of Epochs used for Training',default=10,type=int)
	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parse_args()

	with tf.Session() as sess:
		net = Model(args.batch_size,args.num_epochs)
		net.build_network()
		if not args.forward_only:
			net.fit()
		else:
			net.predict()


