import cv2
import tensorflow as tf
import numpy as np

class Model(object):

	def __init__(self,sess,batch_size,num_epochs,tf_record_file_path):

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.sess = sess
		self.img_height = 480
		self.img_width = 640

		self.tf_record_file_path = tf_record_file_path
		self.filename_queue = tf.train.string_input_producer([self.tf_record_file_path], num_epochs=self.num_epochs)
		self.images, self.labels = self.load_from_tfRecord(self.filename_queue)

	def build_network(self):
		
		print "Building Network"

	def predict(self):

		print "Predicting Model"

	def fit(self):
		print "Training Model"
		
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