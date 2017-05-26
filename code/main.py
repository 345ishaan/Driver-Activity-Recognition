import os
import sys
import argparse
import tensorflow as tf
from model import *


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f','--mode',dest='forward_only',help='Test or Train Mode',default=0,type=int)
	parser.add_argument('-b','--batch_size',dest='batch_size',help='Batch Size for Training',default=32,type=int)
	parser.add_argument('-e','--num_epochs',dest='num_epochs',help='Number of Epochs used for Training',default=1,type=int)
	parser.add_argument('-p1','--tf_train_record_path',dest='tf_train_record_file_path',help='TF Train Record File Path',default=None,type=str)
	parser.add_argument('-p2','--tf_val_record_path',dest='tf_val_record_file_path',help='TF Val Record File Path',default=None,type=str)
	parser.add_argument('-l','--load_model',dest='load_model',help='Flag to Restore previous saved model',default=0,type=int)
	parser.add_argument('-m1','--model_path',dest='model_save_path',help='Checkpoint Location',default=None,type=str)
	parser.add_argument('-m2','--best_model_path',dest='best_model_path',help='Best Checkpoint Location',default=None,type=str)
	parser.add_argument('-m3','--restore_model_path',dest='restore_model_path',help='Restore Checkpoint Location',default=None,type=str)
	parser.add_argument('-t','--write_tensorboard',dest='write_tensorboard_flag',help='Flag for writing in tensorboad',default=True,type=bool)
	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parse_args()
	
	if not os.path.exists('../logs'):
		os.makedirs('../logs')
	if not os.path.exists(args.model_save_path):
		os.makedirs(args.model_save_path)
	if not os.path.exists(args.best_model_path):
		os.makedirs(args.best_model_path)

	map(os.unlink,(os.path.join( '../logs',f) for f in os.listdir('../logs')) )
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		net = Model(sess,args.batch_size,args.num_epochs,args.tf_val_record_file_path,args.tf_train_record_file_path,args.load_model,args.model_save_path, 
			args.best_model_path,args.restore_model_path,args.write_tensorboard_flag)
		net.build_network()
		#net.print_variables()
		if args.write_tensorboard_flag:
			net.write_tensorboard()
		
		if not args.forward_only:
			net.fit()
		else:
			net.predict()


