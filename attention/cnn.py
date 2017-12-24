import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception

def VGG_CAM(input,reuse_flag=False):

	with tf.variable_scope('vgg_cam',reuse=reuse_flag):
		with slim.arg_scope([slim.conv2d,slim.fully_connected],
			activation_fn=tf.nn.relu,
			padding='SAME',
			weights_initializer=tf.truncated_normal_initializer(0.0,0.01)):

			net = slim.repeat(input, 2, slim.conv2d,64,[3, 3], scope='conv1')
			net = slim.max_pool2d(net, [2, 2], scope='pool1')
			
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			net = slim.max_pool2d(net, [2, 2], scope='pool2')
			
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
			net = slim.max_pool2d(net, [2, 2], scope='pool3')
			
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
			net = slim.max_pool2d(net, [2, 2], scope='pool4')
			
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
			net = slim.conv2d(net, 1024, [3,3], scope = 'conv6/CAM_conv')
			
		return net


def inception_v2(inputs,reuse_flag=False):

	with slim.arg_scope(inception.inception_v2_arg_scope()):
			net, end_points = inception.inception_v2(inputs)
			print end_points
	return net,end_points['Mixed_5c']







