import os
import sys
import argparse
import tensorflow as tf
from model import *


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode',dest='forward_only',help='Test or Train Mode',default=0,type=int)
	parser.add_argument('-b','--batch_size',dest='batch_size',help='Batch Size for Training',default=32,type=int)
	parser.add_argument('-e','--num_epochs',dest='num_epochs',help='Number of Epochs used for Training',default=1,type=int)
	parser.add_argument('-p','--tf_record_path',dest='tf_record_file_path',help='TF Record File Path',default=None,type=str)
	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parse_args()
	
	if not os.path.exists('../logs'):
		os.makedirs('../logs')
	map(os.unlink,(os.path.join( '../logs',f) for f in os.listdir('../logs')) )

	with tf.Session() as sess:
		net = Model(sess,args.batch_size,args.num_epochs,args.tf_record_file_path)
		net.build_network()
		if not args.forward_only:
			net.fit()
		else:
			net.predict()


