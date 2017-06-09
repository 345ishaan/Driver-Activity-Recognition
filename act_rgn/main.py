import os
import sys
import argparse
from model import *

def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('-batch',dest='batch_size',help='Batch Size',default=32,type=int)
	parser.add_argument('-epochs',dest='num_epochs',help='Number of Epochs',default=1,type=int)
	parser.add_argument('-img_w',dest='image_width',help='Input Image Width',default=227,type=int)
	parser.add_argument('-img_h',dest='image_height',help='Input Image Height',default=227,type=int)
	parser.add_argument('-img_ch',dest='num_channels',help='Input Image Channels',default=3,type=int)

	parser.add_argument('-cnn_op_dim',dest='spatial_feature_dim',help='CNN Output Feature Spatial Resolution',default=14,type=int)
	parser.add_argument('-cnn_op_depth',dest='spatial_feature_depth',help='CNN Output Features Depth',default=1024,type=int)
	parser.add_argument('-time_length',dest='max_time_steps',help='Maximum Sequence Length of RNN',default=30,type=int)
	parser.add_argument('-tensorboard_flag',dest='write_tensorboard_flag',help='Flag for write mode in tensorboard',default=True,type=bool)
	args = parser.parse_args()
	return args

if os.path.exists('../logs'):
	map(lambda f : os.unlink('../logs/'+f),os.listdir('../logs'))

if __name__ == '__main__':
	args = parse_args()
	
	with tf.Session() as sess:
		model = AttnModel(sess,args)
		model.build_cnn_network()
		model.build_recurrent_model()
		model.load_CNN_weights(scope=None)

		if args.write_tensorboard_flag:
			model.write_tensorboard()
		# model.get_cnn_encodings(d_path='/home/admin/Downloads/ucfTrainTestlist/')
