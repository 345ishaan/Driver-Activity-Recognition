import os
import sys
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# from tensorflow.contrib.slim.nets import vgg
import numpy as np
np.set_printoptions(linewidth=150)
from pdb import set_trace as brk

class Model(object):

	def __init__(self,sess,batch_size,num_epochs,tf_val_record_file_path,tf_train_record_file_path,load_model,model_save_path,best_model_save_path,restore_model_path,write_tensorboard_flag):

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.sess = sess
		
		self.channels = 3

		self.num_classes = 10

		self.tf_train_record_file_path = tf_train_record_file_path
		self.tf_val_record_file_path = tf_val_record_file_path

		self.use_vgg = False
		self.use_resnet_101 = False
		self.use_resnet_50 = False
		self.use_squeezenet = False
		self.use_VGGCAM = True

		if self.use_squeezenet or self.use_VGGCAM:
			self.img_height = 227
			self.img_width = 227
		else:
			self.img_height = 224
			self.img_width = 224

		self.model_save_path = model_save_path
		self.best_model_save_path = best_model_save_path
		self.restore_model_path = restore_model_path

		self.load_model =  load_model
		self.save_after_steps = 200
		self.print_after_steps = 50
		self.perform_val_after_steps = 100

		self.train_data_size = 20000
		self.val_data_size = 2424
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
			print "Using VGG"
			self.net = self.VGG16(self.X,self.keep_prob)
			
		if self.use_resnet_50:
			print "Using Resnet 50"
			self.net = self.resnet_50(self.X,self.keep_prob)
			self.net = tf.squeeze(self.net,axis=[1,2])

		if self.use_resnet_101:
			print "Using Resnet 101"
			self.net = self.resnet_101(self.X,self.keep_prob)
			self.net = tf.squeeze(self.net,axis=[1,2])

		
		if self.use_squeezenet:
			print "Using Squeezenet"
			self.net = self.squeezenet(self.X,self.keep_prob)
			self.net = slim.dropout(self.net,self.keep_prob, scope='dropout_fire9')
			with slim.arg_scope([slim.conv2d],activation_fn = tf.nn.relu,weights_initializer = tf.truncated_normal_initializer(0.0, 0.01)):
				self.net_hm = slim.conv2d(self.net, self.num_classes, [1,1], 1, scope='out_classification')
			self.net = slim.avg_pool2d(self.net_hm,[self.net_hm.get_shape()[1],self.net_hm.get_shape()[2]],stride=[1,1])
			self.net = tf.squeeze(self.net,axis=[1,2])
		
		if self.use_VGGCAM:
			print "Using VGGCAM"
			self.net, self.cam_conv = self.VGG16CAM(self.X)
			self.CAM = self.get_CAM(self.X)


		self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(self.Y,self.num_classes),
			logits=self.net))
		
		# self.minimize_loss = tf.train.AdamOptimizer(0.0001).minimize(self.cross_entropy_loss,var_list=[p for p in tf.trainable_variables() 
		# 	if (('fc_classification' in p.name) or ('out_classification' in p.name))])

		self.minimize_loss = tf.train.MomentumOptimizer(1e-4,0.9,use_nesterov=True).minimize(self.cross_entropy_loss)#,var_list=[p for p in tf.trainable_variables() 
			# if (('fc_classification' in p.name) or ('out_classification' in p.name))])
		#self.minimize_loss = tf.train.AdamOptimizer(1e-7).minimize(self.cross_entropy_loss)		

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.net,1),tf.int32),self.Y),tf.float32))
		
		self.saver = tf.train.Saver(max_to_keep=4,keep_checkpoint_every_n_hours=2)
		self.best_saver = tf.train.Saver(max_to_keep=10)
		
		return self.cross_entropy_loss

	def VGG16(self,inputs,keep_prob):

		with tf.variable_scope('vgg_16'):
			with slim.arg_scope([slim.conv2d, slim.fully_connected],
								 activation_fn = tf.nn.relu,
								 weights_initializer = tf.truncated_normal_initializer(0.0, 0.01),weights_regularizer=slim.l2_regularizer(0.0005)):
				
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


	def resnet_101(self,inputs,keep_prob):

		with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=True)):
			net, end_points = resnet_v1.resnet_v1_101(inputs)

		net = slim.dropout(net,keep_prob, scope='net')

		fc_classification = slim.fully_connected(net,2048,scope='fc_classification')
		fc_classification = slim.dropout(fc_classification,keep_prob, scope='dropout_fc_classification')
		out_classification = slim.fully_connected(fc_classification,self.num_classes,scope='out_classification',activation_fn=None)

		return out_classification

	
	def squeezenet(self, inputs,keep_prob):

		with tf.variable_scope('squeeze_net'):	
			with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu,
									padding = 'SAME',
									weights_initializer = tf.truncated_normal_initializer(0.0, 0.01),weights_regularizer=slim.l2_regularizer(0.0005)):
				
				conv1 = slim.conv2d(inputs, 64, [3,3], 2, padding = 'VALID', scope='conv1')
				pool1 = slim.max_pool2d(conv1, [3,3], 2, scope='pool1')
				fire2 = self.fire_module(pool1, 16, 64, scope = 'fire2')
				fire3 = self.fire_module(fire2, 16, 64, scope = 'fire3', res_connection=False)
				pool3 = slim.max_pool2d(fire3, [3,3], 2, scope='pool3')

				fire4 = self.fire_module(pool3, 32, 128, scope = 'fire4')
				fire5 = self.fire_module(fire4, 32, 128, scope = 'fire5', res_connection=False)
				pool5 = slim.max_pool2d(fire5, [3,3], 2, scope='pool5')

				fire6 = self.fire_module(pool5, 48, 192, scope = 'fire6')
				fire7 = self.fire_module(fire6, 48, 192, scope = 'fire7', res_connection=False)
				fire8 = self.fire_module(fire7, 64, 256, scope = 'fire8')
				fire9 = self.fire_module(fire8, 64, 256, scope = 'fire9', res_connection=False)
				
				
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
		
		with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu,
							padding = 'SAME',
							weights_initializer = tf.truncated_normal_initializer(0.0, 0.01),weights_regularizer=slim.l2_regularizer(0.0005)):
			with tf.variable_scope(scope):
				sq = slim.conv2d(inputs, channels, [1,1], 1, scope = '1x1')
		
		return sq

	def expand(self, inputs, channels, scope):
		
		with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu,
							padding = 'SAME',
							weights_initializer = tf.truncated_normal_initializer(0.0, 0.01),weights_regularizer=slim.l2_regularizer(0.0005)):
			with tf.variable_scope(scope):
				e1x1 = slim.conv2d(inputs, channels, [1,1], 1, scope='1x1')
				e3x3 = slim.conv2d(inputs, channels, [3,3], 1, scope='3x3')
				expand = tf.concat([e1x1, e3x3],3)
		
		return expand

	def VGG16CAM(self,inputs, reuse = False):

		with tf.variable_scope('vgg_cam', reuse = reuse):
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
				net = slim.conv2d(net, 1024, [3,3], scope = 'conv6/CAM_conv')
				cam_conv = net
				net = slim.avg_pool2d(net, [14,14], scope='CAM_pool')
				net = self.flatten(net)
				net = slim.fully_connected(net,self.num_classes, activation_fn = None, scope='out_classification')
		return net, cam_conv


	def flatten(self, inputs):
		with tf.variable_scope('flatten'):
			shape = int(np.prod(inputs.get_shape()[1:]))
			out = tf.reshape(inputs, [-1,shape])
		return out


	def group_conv2d(self,input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, group = 2, scope="conv2d"):
		with tf.variable_scope(scope):
			weights = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1]/group, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
			convolve = lambda i, k : tf.nn.conv2d(i, k, strides=[1, d_h, d_w, 1], padding='SAME')
			input_groups = tf.split(input_, group, 3)
			kernel_groups = tf.split(weights, group, 3)
			output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
			output = tf.concat(output_groups, 3)
			biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.bias_add(output, biases)
		return tf.nn.relu(conv)

	def write_tensorboard(self):

		self.writer = tf.summary.FileWriter('../logs', self.sess.graph)
		loss_summ = tf.summary.scalar('loss', self.cross_entropy_loss)
		img_summ = tf.summary.image('images', self.X,max_outputs=5)
		# self.CAM_summ = tf.summary.image('images_CAM', self.CAM, max_outputs=5)
		label_summ = tf.summary.histogram('labels', self.Y)
		logits_summ = tf.summary.histogram('logits', self.net)
		self.merge_summ = tf.summary.merge([loss_summ,img_summ,label_summ,logits_summ])
		self.writer.flush()

	def predict(self):

		print "Predicting Model"


	def fit_aug(self):
		print "Training Model"

		if self.load_model:
			print "Restoring Model"
			ckpt = tf.train.get_checkpoint_state(self.restore_model_path)
			if ckpt and ckpt.model_checkpoint_path:
				self.saver.restore(self.sess,ckpt.model_checkpoint_path)
				self.sess.run(tf.local_variables_initializer())
		else:
			print "Initializing Model"
			self.sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
			if self.use_vgg:
				self.load_VGG16(path='../weights/vgg16_weights.npz')
			
			if self.use_resnet_50:
				tf.contrib.framework.assign_from_checkpoint_fn('../weights/resnet_v1_50.ckpt',slim.get_model_variables('resnet_v1_50'))(self.sess)
			if self.use_resnet_101:
				tf.contrib.framework.assign_from_checkpoint_fn('../weights/resnet_v1_101.ckpt',slim.get_model_variables('resnet_v1_101'))(self.sess)

			if self.use_squeezenet:
				self.load_squeezenet(path='../weights/sqz_full.mat')
			if self.use_VGGCAM:
				self.load_VGGCAM(path='../weights/vgg16_weights.npz')

		
		counter = 0
		best_val_loss = sys.maxint
		mean_pixel = [103.939, 116.779, 123.68]
		# datagen_train = ImageDataGenerator(
		# 	    width_shift_range=0.,
		# 	    height_shift_range=0.,
		# 	    shear_range=0.,
		# 	    zoom_range=0.,
		# 	    channel_shift_range=0.0,

		# 	    fill_mode='nearest',
		# 	    horizontal_flip=False,
		# 	    rescale=None,
		# 	    preprocessing_function=None)

		datagen_train = ImageDataGenerator()
		datagen_val = ImageDataGenerator()

		train_data_iterator = datagen_train.flow_from_directory('/home/ishan_shashank/state_farm/data_augmentation/new_data/',target_size=(self.img_height, self.img_width),
			batch_size=self.batch_size,class_mode='sparse',shuffle=True)

		val_data_iterator = datagen_val.flow_from_directory('/home/ishan_shashank/state_farm/split_data/validation',target_size=(self.img_height, self.img_width),
			batch_size=self.batch_size,class_mode='sparse')

		try:
			for it in range(20000):
				batch_imgs,batch_labels = train_data_iterator.next()

				if batch_imgs.shape[0] != self.batch_size:
					continue	

				batch_imgs[:,:,:,0] -= mean_pixel[0]
				batch_imgs[:,:,:,1] -= mean_pixel[1]
				batch_imgs[:,:,:,2] -= mean_pixel[2]
					
					
				if self.write_tensorboard_flag:
					__,batch_loss,total_summ = self.sess.run([self.minimize_loss,self.cross_entropy_loss,self.merge_summ],{self.X:batch_imgs,self.Y:batch_labels,self.keep_prob:0.5})
					self.writer.add_summary(total_summ)
				else:
					__,batch_loss = self.sess.run([self.minimize_loss,self.cross_entropy_loss],{self.X:batch_imgs,self.Y:batch_labels,self.keep_prob:0.5})
					
				accuracy = self.sess.run([self.accuracy], feed_dict={self.X:batch_imgs,self.Y:batch_labels,self.keep_prob:1.0})
				
				if counter%self.print_after_steps == 0:
					print "Iteration:{},Loss:{},Training Accuracy:{}".format(counter,batch_loss,accuracy)
				

				if (counter%self.save_after_steps == 0):
					self.saver.save(self.sess,self.model_save_path+'statefarm_model',global_step=int(counter),write_meta_graph=False)
					
				if (counter%self.perform_val_after_steps == 0):
					total_val_acc = 0.0
					total_val_loss = 0.0
					total_val = 0.0
					class_acc ={}
					total_class_acc={}

					total_val_gt_labels = None
					total_val_pred_labels = None

					for itt in range(self.val_data_size/self.batch_size):
						
						batch_val_images,batch_val_labels = val_data_iterator.next()

						if batch_val_images.shape[0] != self.batch_size:
							continue

						if total_val_gt_labels is None:
							total_val_gt_labels = batch_val_labels
						else:
							total_val_gt_labels = np.concatenate([total_val_gt_labels,batch_val_labels],axis=0)
						batch_val_images[:,:,:,0] -= mean_pixel[0]
						batch_val_images[:,:,:,1] -= mean_pixel[1]
						batch_val_images[:,:,:,2] -= mean_pixel[2]
						
						val_logits,val_accuracy,val_loss = self.sess.run([self.net,self.accuracy,self.cross_entropy_loss], feed_dict={self.X:batch_val_images,self.Y:batch_val_labels,
							self.keep_prob:1.0})

		
						
						pred_class = np.argmax(val_logits,axis=1)
						
						if total_val_pred_labels is None:
							total_val_pred_labels = pred_class
						else:
							total_val_pred_labels = np.concatenate([total_val_pred_labels,pred_class],axis=0)

						for k in range(self.num_classes):
							class_acc[k] = class_acc.get(k,0.0) + len(set(np.where(batch_val_labels==k)[0]) & set(np.where(pred_class==k)[0]))
							total_class_acc[k] = total_class_acc.get(k,0.0) + len(np.where(batch_val_labels==k)[0])

						total_val_acc += val_accuracy *(batch_val_images.shape[0])
						total_val_loss += val_loss *(batch_val_images.shape[0])
						total_val += batch_val_images.shape[0]
					total_val_acc = total_val_acc/total_val
					total_val_loss = total_val_loss/total_val

					if self.use_VGGCAM:
						batch_val_images, batch_val_labels = val_data_iterator.next()
						while batch_val_images.shape[0] != self.batch_size:
							batch_val_images, batch_val_labels  = val_data_iterator.next()

						CAM_imgs = self.sess.run(self.CAM, feed_dict={self.X: batch_val_images})
						brk()
						self.save_CAM_images(CAM_imgs, batch_val_images, batch_val_labels)

						# self.writer.add_summary(CAM_summ)



					confusion_matrix = tf.confusion_matrix(np.squeeze(total_val_gt_labels.reshape(-1,1)),np.squeeze(total_val_pred_labels.reshape(-1,1)),num_classes=self.num_classes).eval(session=self.sess)
					
					confusion_matrix = confusion_matrix/np.sum(confusion_matrix,axis=1,keepdims=True).astype(np.float32)


					print "************Validation Accuracy:{},Loss:{}************".format(total_val_acc,total_val_loss)
					for key in class_acc:
						if total_class_acc[key] > 0:
							print "Learning for Class:{},\t{}".format(key,class_acc[key]/total_class_acc[key])
						else:
							print "Learning for Class:{},\t{}".format(key,None)
					print 'Confusion Matrix........'
					print confusion_matrix


					if total_val_loss <= best_val_loss:
						best_val_loss = total_val_loss
						self.best_saver.save(self.sess,self.best_model_save_path+'statefarm_best_model',global_step=int(counter),write_meta_graph=False)
					
				counter += 1
			print 'Training Done'
		except Exception as e:
			print 'Error in Training:{}'.format(e)



	def fit(self):
		print "Training Model"

		if self.load_model:
			print "Restoring Model"
			ckpt = tf.train.get_checkpoint_state(self.restore_model_path)
			if ckpt and ckpt.model_checkpoint_path:
				self.saver.restore(self.sess,ckpt.model_checkpoint_path)
				self.sess.run(tf.local_variables_initializer())
		else:
			print "Initializing Model"
			self.sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
			if self.use_vgg:
				self.load_VGG16(path='../weights/vgg16_weights.npz')
			if self.use_resnet_50:
				tf.contrib.framework.assign_from_checkpoint_fn('../weights/resnet_v1_50.ckpt',slim.get_model_variables('resnet_v1_50'))(self.sess)
			if self.use_resnet_101:
				tf.contrib.framework.assign_from_checkpoint_fn('../weights/resnet_v1_101.ckpt',slim.get_model_variables('resnet_v1_101'))(self.sess)
			if self.use_squeezenet:
				self.load_squeezenet(path='../weights/sqz_full.mat')
			if self.use_VGGCAM:
				self.load_VGGCAM(path='../weights/VGG16CAM.npy')
		

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)

		counter = 0
		best_val_loss = sys.maxint
		mean_pixel = [103.939, 116.779, 123.68]

		try:
			while not coord.should_stop():
				batch_imgs,batch_labels = self.sess.run([self.train_images,self.train_labels])
				

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
					total_val_acc = 0.0
					total_val_loss = 0.0
					total_val = 0.0
					for it in range(self.val_data_size/self.batch_size):

						batch_val_images,batch_val_labels = self.sess.run([self.val_images,self.val_labels])
						batch_val_images[:,:,:,0] -= mean_pixel[0]
						batch_val_images[:,:,:,1] -= mean_pixel[1]
						batch_val_images[:,:,:,2] -= mean_pixel[2]
						
						val_accuracy,val_loss = self.sess.run([self.accuracy,self.cross_entropy_loss], feed_dict={self.X:batch_val_images,self.Y:batch_val_labels,self.keep_prob:1.0})
						total_val_acc += val_accuracy *(batch_val_images.shape[0])
						total_val_loss += val_loss *(batch_val_images.shape[0])
						total_val += batch_val_images.shape[0]
					total_val_acc = total_val_acc/total_val
					total_val_loss = total_val_loss/total_val


					print "************Validation Accuracy:{},Loss:{}************".format(total_val_acc,total_val_loss)
					
					if total_val_loss <= best_val_loss:
						best_val_loss = total_val_loss
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
		print 'Loading Squeezenet weights...'

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

	# def load_VGGCAM(self,path=None):
	# 	weights = np.load(path).item()
	# 	params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_cam')
	# 	for var in params:
	# 		if 'Momentum' in var.name:
	# 			continue
	# 		print var.name
	# 		spl = var.name.split('/')
	# 		i = spl[2]
	# 		j = spl[3][:-2]
	# 		self.sess.run(var.assign(weights[i][j]))
		

	def load_VGGCAM(self,path=None):
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_cam')
		weights = np.load(path)
		keys = sorted(weights.keys())
		print 'Loading VGG16 weights...'
		for var, k in zip(variables, keys):
			if (('conv'  in k) and ('conv6' not in k)):
				self.sess.run(var.assign(weights[k]))


	def get_CAM(self, image):
		with tf.name_scope('get_CAM'):
			weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_cam/out_classification/weights')
			convCAM_res = tf.image.resize_images(self.cam_conv, [self.img_width, self.img_height])
			mask = tf.nn.conv2d(convCAM_res, tf.expand_dims(weights, axis=0), strides=[1, 1, 1, 1], padding='VALID')
			return mask



	def print_variables(self):

		params = tf.trainable_variables()#slim.get_model_variables()#tf.all_variables()
		for param in params:
			print param.name, ' ', param.get_shape()


	def save_CAM_images(self, CAM_imgs, orig_images, labels):
		num_imgs = orig_images.shape[0]
		for i in xrange(10):
			if not os.path.exists('../CAM_images/%d' % i):
				os.makedirs('../CAM_images/%d' % i)

		for i in xrange(num_imgs):
			for j in xrange(10):
				plt.imshow(orig_images[i,:,:,:]/255.0)
				plt.imshow(CAM_imgs[i,:,:,j]/255.0,cmap="jet",alpha=0.5, interpolation= 'nearest')
				plt.savefig('../CAM_images/%d/im%d_ch%d.png' % (labels[i],i,j))
				plt.close('all')

