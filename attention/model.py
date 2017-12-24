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
		self.num_classes = args.num_classes

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
		self.cnn_ip = tf.placeholder(tf.float32,shape=(self.batch_size,self.image_height,self.image_width,self.num_channels))
		if self.use_inception:
			__,self.cnn_op = inception_v2(self.cnn_ip,reuse_flag=False)
			print self.cnn_op.get_shape()
		if self.use_VGG:
			self.cnn_op = VGG_CAM(self.cnn_ip,reuse_flag=False)
		print "Built CNN Network"



	def build_recurrent_model(self):

		print 'Building Recurrent Model....'

		self.conv_2_rnn_ip = tf.placeholder(tf.float32,shape=(self.max_time_steps,self.batch_size,self.num_spatial_locations,self.spatial_feature_depth))
		self.labels = tf.placeholder(tf.int32,shape=(self.max_time_steps,self.batch_size))

		mean_ip = tf.reduce_sum(tf.reduce_sum(self.conv_2_rnn_ip,0),1)
		
		with tf.variable_scope('init_scheme') as scope:
			self.mean_to_space_w = tf.get_variable(name='m_to_s_w',shape=(self.spatial_feature_depth,self.spatial_feature_depth),dtype=tf.float32)
			self.mean_to_space_b = tf.get_variable(name='m_to_s_b',shape=(self.spatial_feature_depth),dtype=tf.float32)
			
			self.init_memory_w = tf.get_variable(name='m_w',shape=(self.spatial_feature_depth,self.lstm_hidden_dim),dtype=tf.float32)
			self.init_memory_b = tf.get_variable(name='m_b',shape=(self.lstm_hidden_dim),dtype=tf.float32)
			
			self.init_output_w = tf.get_variable(name='o_w',shape=(self.spatial_feature_depth,self.lstm_hidden_dim),dtype=tf.float32)
			self.init_output_b = tf.get_variable(name='o_b',shape=(self.lstm_hidden_dim),dtype=tf.float32)

		with tf.variable_scope('predictions') as scope:
			self.logits_w = tf.get_variable(name='logits_w',shape=(self.lstm_hidden_dim,self.num_classes),dtype=tf.float32)
			self.logits_b = tf.get_variable(name='logits_b',shape=(self.num_classes),dtype=tf.float32)

		self.mean_ip = tf.tanh(tf.matmul(mean_ip,self.mean_to_space_w) + self.mean_to_space_b)
		self.attn_init_output = tf.tanh(tf.matmul(self.mean_ip,self.init_output_w) + self.init_output_b)

		cells=[]

		for i in range(self.num_layers):
			cell = rnn.BasicLSTMCell(self.lstm_hidden_dim,reuse=None)
			cells.append(cell)

		self.cell = rnn.MultiRNNCell(cells,state_is_tuple=True)

		self.initial_state = self.cell.zero_state(self.batch_size,tf.float32) # Gives the initial state of the cell and output for all layers

		decoder_inputs = map(lambda x:tf.squeeze(x),tf.split(self.conv_2_rnn_ip, self.max_time_steps,axis=0))
		
		outputs,state = actrgn_rnn_decoder(decoder_inputs, self.initial_state, self.attn_init_output,self.cell,self.attn_dim,self.lstm_hidden_dim)
		
		initializer = tf.constant(0.0)
		self.counter = 0
		def acc(prev_loss,cur_input):
			
			logit = tf.tanh(tf.matmul(outputs[self.counter],self.logits_w) + self.logits_b)
			loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(cur_input,self.num_classes),logits=logit)))
			self.counter += 1
			return loss

		map_op = tf.scan(acc,self.labels,initializer=initializer,name='compute_loss')		
		
		total_loss = tf.reduce_sum(tf.reduce_mean(map_op))
		self.optimizer = tf.train.AdamOptimizer().minimize(total_loss)

		print 'Done Building Recurrent Model'



	def write_tensorboard(self):
		self.writer = tf.summary.FileWriter('../logs', self.sess.graph)
		self.writer.flush()

	def print_vars(self,scope=None):
		
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

	def load_inception_v2(self):
		to_load=[]
		for param in slim.get_model_variables('InceptionV2'):
			if 'Logits' not in param.name:
				to_load.append(param)
		tf.contrib.framework.assign_from_checkpoint_fn('../weights/inception_v2.ckpt',to_load)(self.sess)

	def get_cnn_encodings(self,image):

		input_feed={self.cnn_ip:image}
		return self.sess.run(self.cnn_op,input_feed)


	def create_hdf5(self,path):
		
		f = h5py.File('cnn_encodings.hdf5','w')
		
		df = np.asarray(pd.read_csv(path+'data_file.csv'))
		df = df[np.where(df[:,0]=='train')]
		total_frames  = np.sum(df[:,3].astype(np.int32))

		image_set = f.create_dataset('features',(total_frames,240,320,3),dtype='float32',shuffle=False,compression="gzip")
		
		mean_pixel = [103.939, 116.779, 123.68]
		count  = 0 
		for i in tqdm(xrange(df.shape[0])):
			for j in xrange(int(df[i,3])):
				path = path+'train/'+str(df[i,1])+'/'+str(df[i,2])+'-'+'0'*(4-len(str(j+1)))+str(j+1)+'.jpg'

				image = cv2.imread(path).astype(np.float32)
				image = np.expand_dims(image, axis=0)
				image[:,:,:,0] -= mean_pixel[0]
				image[:,:,:,1] -= mean_pixel[1]
				image[:,:,:,2] -= mean_pixel[2]
				image = tf.image.resize_images(image, [224,224]).eval(session=self.sess)
				image_set[count,:,:,:] = get_cnn_encodings(image)
				count += 1
			
		print "here"



		


		
















