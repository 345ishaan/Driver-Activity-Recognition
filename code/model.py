import os
import sys
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
# from tensorflow.contrib.slim.nets import vgg
import numpy as np
from pdb import set_trace as brk

class Model(object):

	def __init__(self,sess,batch_size,num_epochs,tf_val_record_file_path,tf_train_record_file_path,load_model,model_save_path,best_model_save_path,restore_model_path,write_tensorboard_flag):

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.sess = sess
		self.img_height = 224
		self.img_width = 224
		self.channels = 3

		self.num_classes = 10

		self.tf_train_record_file_path = tf_train_record_file_path
		self.tf_val_record_file_path = tf_val_record_file_path

		self.use_vgg = False
		self.use_resnet = False
		self.use_squeezenet = True
		
		self.model_save_path = model_save_path
		self.best_model_save_path = best_model_save_path
		self.restore_model_path = restore_model_path

		self.load_model =  load_model
		self.save_after_steps = 200
		self.print_after_steps = 50
		self.perform_val_after_steps = 100

		self.train_data_size = 19714
		self.val_data_size = 2710
		self.it_per_epoch = self.train_data_size/float(self.batch_size)

		self.write_tensorboard_flag = write_tensorboard_flag

		print "Batch Size:{},Number of Epochs:{}".format(self.batch_size,self.num_epochs)
		
	def build_network(self):

		print "Building Network"
		with tf.variable_scope('train_data_queue'):
			filename_train_queue = tf.train.string_input_producer([self.tf_train_record_file_path], num_epochs=self.num_epochs)
			self.train_images, self.train_labels = self.load_from_tfRecord(filename_train_queue)
		with tf.variable_scope('val_data_queue'):
			filename_val_queue = tf.train.string_input_producer([self.tf_val_record_file_path], 
				num_epochs=self.num_epochs*(int(self.it_per_epoch/self.perform_val_after_steps)+1))
			self.val_images, self.val_labels = self.load_from_tfRecord(filename_val_queue)


		self.X = tf.placeholder(tf.float32,shape=(self.batch_size,self.img_height,self.img_width,self.channels),name='input')
		self.Y = tf.placeholder(tf.int32,shape=(self.batch_size),name='labels')
		self.keep_prob = tf.placeholder(tf.float32,shape=[],name='dropout_prob')

		if self.use_vgg:
			self.net = self.VGG16(self.X,self.keep_prob)
			
		if self.use_resnet:
			self.net = self.resnet_50(self.X,self.keep_prob)
			self.net = tf.squeeze(self.net,axis=[1,2])
		if self.use_squeezenet:
			self.net = self.squeezenet(self.X,self.keep_prob)
			self.net = slim.conv2d(self.net, self.num_classes, [1,1], 1, scope='out_classification')
			self.net = slim.avg_pool2d(self.net,[self.net.get_shape()[1],self.net.get_shape()[2]],stride=[1,1])
			self.net = tf.squeeze(self.net,axis=[1,2])
			
		self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(self.Y,self.num_classes),
			logits=self.net))
		
		self.minimize_loss = tf.train.AdamOptimizer().minimize(self.cross_entropy_loss,var_list=[p for p in tf.trainable_variables() 
			if (('fc_classification' in p.name) or ('out_classification' in p.name))])
		
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.net,1),tf.int32),self.Y),tf.float32))
		self.saver = tf.train.Saver(max_to_keep=4,keep_checkpoint_every_n_hours=2)
		self.best_saver = tf.train.Saver(max_to_keep=10)
		
		return self.cross_entropy_loss

	def VGG16(self,inputs,keep_prob):

		with tf.variable_scope('vgg_16'):
			with slim.arg_scope([slim.conv2d, slim.fully_connected],
								 activation_fn = tf.nn.relu,
								 weights_initializer = tf.truncated_normal_initializer(0.0, 0.01)):
				
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
				net = slim.fully_connected(net, 4096, scope='fc_classification')
				net = slim.dropout(net,keep_prob, scope='dropout_fc_classification')
				net = slim.fully_connected(net,self.num_classes,scope='out_classification', activation_fn=None)
		return net

	def resnet_50(self,inputs,keep_prob):

		with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=True)):
			net, end_points = resnet_v1.resnet_v1_50(inputs)

		net = slim.dropout(net,keep_prob, scope='net')
		fc_classification = slim.fully_connected(net,2048,scope='fc_classification')
		fc_classification = slim.dropout(fc_classification,keep_prob, scope='dropout_fc_classification')
		out_classification = slim.fully_connected(fc_classification,self.num_classes,scope='out_classification',activation_fn=None)

		return out_classification

	
	def squeezenet(self, inputs,keep_prob):

		with tf.variable_scope('squeeze_net'):	
			with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu,
									padding = 'SAME',
									weights_initializer = tf.truncated_normal_initializer(0.0, 0.01)):

				conv1 = slim.conv2d(inputs, 64, [3,3], 2, padding = 'VALID', scope='conv1')
				pool1 = slim.max_pool2d(conv1, [2,2], 2, scope='pool1')
				fire2 = self.fire_module(pool1, 16, 64, scope = 'fire2')
				fire3 = self.fire_module(fire2, 16, 64, scope = 'fire3', res_connection=True)
				fire4 = self.fire_module(fire3, 32, 128, scope = 'fire4')
				pool4 = slim.max_pool2d(fire4, [2,2], 2, scope='pool4')
				fire5 = self.fire_module(pool4, 32, 128, scope = 'fire5', res_connection=True)
				fire6 = self.fire_module(fire5, 48, 192, scope = 'fire6')
				fire7 = self.fire_module(fire6, 48, 192, scope = 'fire7', res_connection=True)
				fire8 = self.fire_module(fire7, 64, 256, scope = 'fire8')
				pool8 = slim.max_pool2d(fire8, [2,2], 2, scope='pool8')
				fire9 = self.fire_module(pool8, 64, 256, scope = 'fire9', res_connection=True)
				
				
		return fire9

	def fire_module(self, inputs, s_channels, e_channels, scope, res_connection = False):
		
		with tf.variable_scope(scope):
			sq = self.squeeze(inputs, s_channels, 'squeeze')
			ex = self.expand(sq, e_channels, 'expand')
			if res_connection:
				ret = tf.nn.relu(tf.add(inputs,ex))
			else:
				ret = tf.nn.relu(ex)
		
		return ret


	def squeeze(self, inputs, channels, scope):
		
		with slim.arg_scope([slim.conv2d], activation_fn = None,
							padding = 'SAME',
							weights_initializer = tf.truncated_normal_initializer(0.0, 0.01)):
			with tf.variable_scope(scope):
				sq = slim.conv2d(inputs, channels, [1,1], 1, scope = '1x1')
		
		return sq

	def expand(self, inputs, channels, scope):
		
		with slim.arg_scope([slim.conv2d], activation_fn = None,
							padding = 'SAME',
							weights_initializer = tf.truncated_normal_initializer(0.0, 0.01)):
			with tf.variable_scope(scope):
				e1x1 = slim.conv2d(inputs, channels, [1,1], 1, scope='1x1')
				e3x3 = slim.conv2d(inputs, channels, [3,3], 1, scope='3x3')
				expand = tf.concat([e1x1, e3x3],3)
		
		return expand



	def write_tensorboard(self):

		self.writer = tf.summary.FileWriter('../logs', self.sess.graph)
		loss_summ = tf.summary.scalar('loss', self.cross_entropy_loss)
		img_summ = tf.summary.image('images', self.X,max_outputs=5)
		label_summ = tf.summary.histogram('labels', self.Y)
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
				self.sess.run(tf.local_variables_initializer())
		else:
			print "Initializing Model"
			self.sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
			if self.use_vgg:
				self.load_VGG16(path='../weights/vgg16_weights.npz')
			if self.use_resnet:
				tf.contrib.framework.assign_from_checkpoint_fn('../weights/resnet_v1_50.ckpt',slim.get_model_variables('resnet_v1_50'))(self.sess)
			if self.use_squeezenet:
				self.load_squeezenet(path='../weights/sqz_full.mat')
		

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)

		counter = 0
		best_val_acc = -sys.maxint

		try:
			while not coord.should_stop():
				batch_imgs,batch_labels = self.sess.run([self.train_images,self.train_labels])
				mean_pixel = [103.939, 116.779, 123.68]

				batch_imgs[:,:,:,0] -= mean_pixel[0]
				batch_imgs[:,:,:,1] -= mean_pixel[1]
				batch_imgs[:,:,:,2] -= mean_pixel[2]
				
				
				if self.write_tensorboard_flag:
					__,batch_loss,total_summ = self.sess.run([self.minimize_loss,self.cross_entropy_loss,self.merge_summ],{self.X:batch_imgs,self.Y:batch_labels,self.keep_prob:0.5})
					
					self.writer.add_summary(total_summ)
				else:
					__,batch_loss = self.sess.run([self.minimize_loss,self.cross_entropy_loss],{self.X:batch_imgs,self.Y:batch_labels,self.keep_prob:0.5})
					
				accuracy = self.sess.run(self.accuracy, feed_dict={self.X:batch_imgs,self.Y:batch_labels,self.keep_prob:1.0})
				
				if counter%self.print_after_steps == 0:
					print "Iteration:{},Loss:{},Training Accuracy:{}".format(counter,batch_loss,accuracy)
				

				if (counter%self.save_after_steps == 0):
					self.saver.save(self.sess,self.model_save_path+'statefarm_model',global_step=int(counter),write_meta_graph=False)
				
				if (counter%self.perform_val_after_steps == 0):
					val_acc = 0.0
					total_val = 0.0
					for it in range(self.val_data_size/self.batch_size):
						batch_val_images,batch_val_labels = self.sess.run([self.val_images,self.val_labels])
						
						accuracy = self.sess.run(self.accuracy, feed_dict={self.X:batch_val_images,self.Y:batch_val_labels,self.keep_prob:1.0})
						val_acc += accuracy *(batch_val_images.shape[0])
						total_val += batch_val_images.shape[0]
					val_acc = val_acc/total_val

					print "************Validation Accuracy:{}************".format(val_acc)
					
					if val_acc >= best_val_acc:
						best_val_acc = val_acc
						self.best_saver.save(self.sess,self.best_model_save_path+'statefarm_best_model',global_step=int(counter),write_meta_graph=False)
					
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
			if 'fc8' not in k:
				self.sess.run(var.assign(weights[k]))

	def load_squeezenet(self,path=None):

		weights = scipy.io.loadmat(path)
		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='squeeze_net')
		
		for param in params:
			print param.name
			splits = param.name.split('/')
			if 'fire' in splits[1]:
				name = splits[1]+'/'+splits[2]+splits[3]
			elif 'conv' in splits[1]:
				name = splits[1]

			if splits[-1].find('weights') != -1:
				self.sess.run(param.assign(weights[name][0,0]))
				
			if splits[-1].find('biases') != -1:
				self.sess.run(param.assign(weights[name][0,1][0]))

	def print_variables(self):

		params = tf.trainable_variables()#slim.get_model_variables()#tf.all_variables()
		for param in params:
			print param.name, ' ', param.get_shape()
							
