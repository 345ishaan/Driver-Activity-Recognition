import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import numpy as np

class Model(object):

	def __init__(self,sess,batch_size,num_epochs,tf_record_file_path):

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.sess = sess
		self.img_height = 480
		self.img_width = 640
		self.channels = 3

		self.tf_record_file_path = tf_record_file_path
		self.filename_queue = tf.train.string_input_producer([self.tf_record_file_path], num_epochs=self.num_epochs)
		self.images, self.labels = self.load_from_tfRecord(self.filename_queue)

	def build_network(self):

		print "Building Network"
		
		self.X = tf.placeholder(tf.float32,shape=(self.batch_size,self.img_height,self.img_width,self.channels),name='input')
		self.Y = tf.placeholder(tf.int32,shape=(self.batch_size),name='labels')

		with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=True)):
			self.net, self.end_points = resnet_v1.resnet_v1_50(self.X, 20)
		return self.net

	def write_tensorboard(self):

		self.writer = tf.summary.FileWriter('../logs', self.sess.graph)
		self.writer.flush()

	def predict(self):

		print "Predicting Model"

	def fit(self):
		print "Training Model"
		
		optimizer = tf.train.AdamOptimizer()
		minimize_loss = optimizer.minimize(tf.losses.softmax_cross_entropy(tf.one_hot(self.Y),tf.squeeze(self.net)))
		
		self.sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
		
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
		try:
			while not coord.should_stop():
				batch_imgs,batch_labels = self.sess.run([self.images,self.labels])
				
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

		resized_image = tf.image.resize_image_with_crop_or_pad(image_tf,target_height=self.img_height,target_width=self.img_width)
		
		images,annotations = tf.train.shuffle_batch([resized_image,labels],batch_size=self.batch_size,num_threads=1,capacity=2000,min_after_dequeue=1000)
		
		return images,annotations

	def print_variables(self):

		params = slim.get_model_variables()

		for param in params:
			print '{} {}'.format(param.name,param.get_shape())