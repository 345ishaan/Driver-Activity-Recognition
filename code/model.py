import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from tensorflow.contrib.slim.nets import resnet_v1
# from tensorflow.contrib.slim.nets import vgg
import numpy as np
from pdb import set_trace as brk

class Model(object):

	def __init__(self,sess,batch_size,num_epochs,tf_record_file_path,load_model,model_save_path,write_tensorboard_flag):

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.sess = sess
		self.img_height = 224
		self.img_width = 224
		self.channels = 3

		self.num_classes = 10

		self.tf_record_file_path = tf_record_file_path
		self.filename_queue = tf.train.string_input_producer([self.tf_record_file_path], num_epochs=self.num_epochs)
		self.images, self.labels = self.load_from_tfRecord(self.filename_queue)
		
		self.model_save_path = model_save_path
		self.load_model =  load_model
		self.save_after_steps = 200
		self.print_after_steps = 50

		self.write_tensorboard_flag = write_tensorboard_flag

		print "Batch Size:{},Number of Epochs:{}".format(self.batch_size,self.num_epochs)
		
	def build_network(self):

		print "Building Network"
		
		self.X = tf.placeholder(tf.float32,shape=(self.batch_size,self.img_height,self.img_width,self.channels),name='input')
		self.Y = tf.placeholder(tf.int32,shape=(self.batch_size),name='labels')
		self.keep_prob = tf.placeholder(tf.float32,shape=[],name='dropout_prob')
		# with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=True)):
		# 	self.net, self.end_points = resnet_v1.resnet_v1_50(self.X)
		# self.fc_classification = slim.fully_connected(self.net,512,scope='fc_classification')
		# self.out_classification = slim.fully_connected(self.fc_classification,self.num_classes,scope='out_classification')
		# self.out_classification,__ = vgg.vgg_16(self.X,num_classes=10,spatial_squeeze=False)
		self.net = self.VGG16(self.X,self.keep_prob)
		

		self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(self.Y,self.num_classes),
			logits=self.net))
		
		self.minimize_loss = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.cross_entropy_loss)

		self.saver = tf.train.Saver(max_to_keep=4,keep_checkpoint_every_n_hours=2)

		
		return self.cross_entropy_loss

	def VGG16(self,inputs,keep_prob):

		with tf.variable_scope('vgg_16'):
			with slim.arg_scope([slim.conv2d, slim.fully_connected],
								 activation_fn = tf.nn.relu,
								 weights_initializer = tf.constant_initializer(0.0)):
				
				net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				shape = int(np.prod(net.get_shape()[1:]))

				net = slim.fully_connected(tf.reshape(net, [-1, shape]), 4096, scope='fc6')
				net = slim.dropout(net,keep_prob, scope='dropout6')
				net = slim.fully_connected(net, 1024, scope='fc7')
				net = slim.dropout(net,keep_prob, scope='dropout7')
				net = slim.fully_connected(net,self.num_classes,scope='classification', activation_fn=None)
		return net

	def write_tensorboard(self):

		self.writer = tf.summary.FileWriter('../logs', self.sess.graph)
		loss_summ = tf.summary.scalar('loss', self.cross_entropy_loss)
		img_summ = tf.summary.image('images', self.images,max_outputs=5)
		label_summ = tf.summary.histogram('labels', self.labels)
		self.merge_summ = tf.summary.merge([loss_summ,img_summ,label_summ])
		self.writer.flush()

	def predict(self):

		print "Predicting Model"

	def fit(self):
		print "Training Model"

		if self.load_model:
			print "Restoring Model"
			ckpt = tf.train.get_checkpoint_state(self.model_save_path)
			if ckpt and ckpt.model_checkpoint_path:
				self.saver.restore(self.sess,ckpt.model_checkpoint_path)
		else:
			print "Initializing Model"
			self.sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
			

		#tf.contrib.framework.assign_from_checkpoint_fn('../weights/resnet_v1_50.ckpt',slim.get_model_variables('resnet_v1_50'))(self.sess)
		
		self.load_VGG16(path='../weights/vgg16_weights.npz')

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)

		counter = 0
		best_loss = sys.maxint

		try:
			while not coord.should_stop():
				batch_imgs,batch_labels = self.sess.run([self.images,self.labels])
				
				mean_pixel = [103.939, 116.779, 123.68]

				batch_imgs[:,:,:,0] -= mean_pixel[0]
				batch_imgs[:,:,:,1] -= mean_pixel[1]
				batch_imgs[:,:,:,2] -= mean_pixel[2]
				
				
				if self.write_tensorboard_flag:
					__,batch_loss,total_summ = self.sess.run([self.minimize_loss,self.cross_entropy_loss,self.merge_summ],{self.X:batch_imgs,self.Y:batch_labels,self.keep_prob:0.5})
					self.writer.add_summary(total_summ)
				else:
					__,batch_loss = self.sess.run([self.minimize_loss,self.cross_entropy_loss],{self.X:batch_imgs,self.Y:batch_labels,self.keep_prob:0.5})
					

				if (counter%self.save_after_steps == 0 or batch_loss <= best_loss):
					best_loss = batch_loss
					self.saver.save(self.sess,self.model_save_path+'statefarm_model',global_step=int(counter),write_meta_graph=False)
				
				if counter%self.print_after_steps == 0:
					print "Iteration:{},Loss:{}".format(counter,batch_loss)
				counter += 1
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')
		finally:
			coord.request_stop()
		coord.join(threads)		
		


	def load_from_tfRecord(self,filename_queue):
		
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		
		features = tf.parse_single_example(
			serialized_example,
			features={
				'image_raw':tf.FixedLenFeature([], tf.string),
				'width': tf.FixedLenFeature([], tf.int64),
				'height':tf.FixedLenFeature([], tf.int64),
				'class': tf.FixedLenFeature([], tf.int64)
			})
		
		image = tf.decode_raw(features['image_raw'], tf.uint8)
		height = tf.cast(features['height'], tf.int32)
		width = tf.cast(features['width'], tf.int32)
		labels = tf.cast(features['class'], tf.int32)

		image_shape = tf.stack([height,width,3])
		image_tf = tf.reshape(image,image_shape)

		#resized_image = tf.image.resize_image_with_crop_or_pad(image_tf,target_height=self.img_height,target_width=self.img_width)
		resized_image = tf.image.resize_images(image_tf,[self.img_height, self.img_width])
		
		images,annotations = tf.train.shuffle_batch([resized_image,labels],batch_size=self.batch_size,num_threads=1,capacity=2000,min_after_dequeue=1000)
		
		return images,annotations

	def load_VGG16(self, path=None):
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16')
		weights = np.load(path)
		keys = sorted(weights.keys())
		print 'Loading VGG16 weights...'
		for var, k in zip(variables, keys):
			if 'conv' in k:
				self.sess.run(var.assign(weights[k]))

	def print_variables(self):

		params = tf.trainable_variables()#slim.get_model_variables()#tf.all_variables()
		for param in params:
			print '{} {}'.format(param.name,param.get_shape())
			
