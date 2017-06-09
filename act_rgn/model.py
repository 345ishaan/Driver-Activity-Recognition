import os
import sys
import tensorflow as tf
from cnn import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import BasicLSTMCell
from tensorflow.contrib import rnn
from ops import *
from ipdb import set_trace as brk
import shutil

class AttnModel(object):

	def __init__(self,sess,args):

		self.sess = sess

		self.num_spatial_locations = args.spatial_feature_dim
		self.spatial_feature_depth = args.spatial_feature_depth

		
		self.batch_size = args.batch_size
		self.num_epochs = args.num_epochs

		self.image_width = args.image_width
		self.image_height = args.image_height
		self.num_channels = args.num_channels

		self.write_tensorboard_flag = args.write_tensorboard_flag

		self.use_VGG = False
		self.use_inception = True

		''' Recurrent Model'''
		self.num_layers = 3
		self.lstm_hidden_dim = 1000
		self.attn_dim= 512
		self.max_time_steps = args.max_time_steps

		print "Initialised Attention Model"

	def build_cnn_network(self):

		#self.X = tf.placeholder(tf.float32,shape=(self.batch_size,self.total_time_steps,self.spatial_feature_dim,self.spatial_feature_dim,self.spatial_feature_depth),name='ip_rnn_seq')
		self.cnn_ip = tf.placeholder(tf.float32,shape=(self.batch_size,self.image_width,self.image_height,self.num_channels))
		if self.use_inception:
			__,self.cnn_op = inception_v4(self.cnn_ip,reuse_flag=False)
			print self.cnn_op.get_shape()
		if self.use_VGG:
			self.cnn_op = VGG_CAM(self.cnn_ip,reuse_flag=False)
		print "Built CNN Network"



	def build_recurrent_model(self):

		print 'Building Recurrent Model....'

		self.conv_2_rnn_ip = tf.placeholder(tf.float32,shape=(self.max_time_steps,self.batch_size,self.num_spatial_locations,self.spatial_feature_depth))
		self.labels = tf.placeholder(tf.int32,shape=(self.batch_size,self.max_time_steps))

		mean_ip = tf.reduce_sum(tf.reduce_sum(self.conv_2_rnn_ip,0),1)
		
		with tf.variable_scope('init_scheme') as scope:
			self.mean_to_space_w = tf.get_variable(name='m_to_s_w',shape=(self.spatial_feature_depth,self.spatial_feature_depth),dtype=tf.float32)
			self.mean_to_space_b = tf.get_variable(name='m_to_s_b',shape=(self.spatial_feature_depth),dtype=tf.float32)
			
			self.init_memory_w = tf.get_variable(name='m_w',shape=(self.spatial_feature_depth,self.lstm_hidden_dim),dtype=tf.float32)
			self.init_memory_b = tf.get_variable(name='m_b',shape=(self.lstm_hidden_dim),dtype=tf.float32)
			
			self.init_output_w = tf.get_variable(name='o_w',shape=(self.spatial_feature_depth,self.lstm_hidden_dim),dtype=tf.float32)
			self.init_output_b = tf.get_variable(name='o_b',shape=(self.lstm_hidden_dim),dtype=tf.float32)


		self.mean_ip = tf.matmul(mean_ip,self.mean_to_space_w) + self.mean_to_space_b
		self.attn_init_output = tf.matmul(self.mean_ip,self.init_output_w) + self.init_output_b


		
		cells=[]

		for i in range(self.num_layers):
			cell = rnn.BasicLSTMCell(self.lstm_hidden_dim,reuse=None)
			cells.append(cell)

		self.cell = rnn.MultiRNNCell(cells,state_is_tuple=True)

		self.initial_state = self.cell.zero_state(self.batch_size,tf.float32) # Gives the initial state of the cell and output for all layers

		

		decoder_inputs = map(lambda x:tf.squeeze(x),tf.split(self.conv_2_rnn_ip, self.max_time_steps,axis=0))
		
		outputs,state = actrgn_rnn_decoder(decoder_inputs, self.initial_state, self.attn_init_output,self.cell,self.attn_dim,self.lstm_hidden_dim)




		# with tf.name_scope('attn_lstm1'):
		# 	self.h1 = tf.tile(tf.Variable(initial_value=tf.zeros([1,self.lstm_hidden_dim],dtype=tf.float32),name='output_state'),[self.batch_size,1])
		# 	self.c1 = tf.tile(tf.Variable(initial_value=tf.zeros([1,self.lstm_hidden_dim],dtype=tf.float32),name='cell_state'),[self.batch_size,1])
		# 	self.state_1 = tf.tuple([self.h1,self.c1],name='lstm_state')
		
		# with tf.name_scope('attn_lstm2'):
		# 	self.h2 = tf.tile(tf.Variable(initial_value=tf.zeros([1,self.lstm_hidden_dim],dtype=tf.float32),name='output_state'),[self.batch_size,1])
		# 	self.c2 = tf.tile(tf.Variable(initial_value=tf.zeros([1,self.lstm_hidden_dim],dtype=tf.float32),name='cell_state'),[self.batch_size,1])
		# 	self.state_2 = tf.tuple([self.h2,self.c2],name='lstm_state')

		# with tf.name_scope('attn_lstm3'):
		# 	self.h3 = tf.tile(tf.Variable(initial_value=tf.zeros([1,self.lstm_hidden_dim],dtype=tf.float32),name='output_state'),[self.batch_size,1])
		# 	self.c3 = tf.tile(tf.Variable(initial_value=tf.zeros([1,self.lstm_hidden_dim],dtype=tf.float32),name='cell_state'),[self.batch_size,1])
		# 	self.state_3 = tf.tuple([self.h3,self.c3],name='lstm_state')

		# with tf.variable_scope('lstm_1'):
		# 	self.lstm1_weights = tf.get_variable('weights',shape=[self.conv_2_rnn_ip.shape[1]+self.lstm_hidden_dim,self.lstm_hidden_dim*4],dtype=tf.float32)
		# 	self.lstm1_biases = tf.get_variable('biases',shape=[4*self.lstm_hidden_dim],dtype=tf.float32)

		# with tf.variable_scope('lstm_2'):
		# 	self.lstm2_weights = tf.get_variable('weights',shape=[2*self.lstm_hidden_dim,self.lstm_hidden_dim*4],dtype=tf.float32)
		# 	self.lstm2_biases = tf.get_variable('biases',shape=[4*self.lstm_hidden_dim],dtype=tf.float32)

		# with tf.variable_scope('lstm_3'):
		# 	self.lstm3_weights = tf.get_variable('weights',shape=[2*self.lstm_hidden_dim,self.lstm_hidden_dim*4],dtype=tf.float32)
		# 	self.lstm3_biases = tf.get_variable('biases',shape=[4*self.lstm_hidden_dim],dtype=tf.float32)

		
		# with tf.variable_scope('lstm_1') as scope:
		# 	self.h1,self.state_1 = self.lstm_cell1(self.conv_2_rnn_ip,self.state_1,scope=scope)
		
		# with tf.variable_scope('lstm_2') as scope:
		# 	self.h2,self.state_2 = self.lstm_cell2(self.h1,self.state_2,scope=scope)
			
		# with tf.variable_scope('lstm_3') as scope:
		# 	self.h3,self.state_3 = self.lstm_cell3(self.h2,self.state_3,scope=scope)
			

		print 'Done Building Recurrent Model'



	def write_tensorboard(self):
		self.writer = tf.summary.FileWriter('../logs', self.sess.graph)
		self.writer.flush()

	def load_CNN_weights(self,scope=None):
		
		params = None
		if scope is None:
			params = tf.trainable_variables()
		else:
			params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
		for param in params:
			print param.name,param.get_shape()


	def load_VGG16_imagenet(self, path=None):
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_cam')
		weights = np.load(path)
		keys = sorted(weights.keys())
		print 'Loading VGG16 weights...'
		for var, k in zip(variables, keys):
			self.sess.run(var.assign(weights[k]))

	def get_cnn_encodings(self,path=None):

		class_map = {}

		fp = open(f_path+'class_map.txt','rb')
		
		for row in fp:
			id_,class_ = row.split(' ')
			class_map[class_] = int(id_)

		fp.close()

		dirs = os.listdir(path+'/videos/')
		
		for d in dirs:
			if not os.path.exists(path+'/cnn_embeddings/'+d):
				os.makedirs(path+'/cnn_embeddings/'+d)
			for vid in os.listdir(d):
				os.system('ffmpeg -i {} -r 30 filename%03d.jpg'.format(vid))


		
















