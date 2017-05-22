import tensorflow as tf


class Model(object):

	def __init__(self,batch_size,num_epochs):

		self.batch_size = batch_size
		self.num_epochs = num_epochs

	def build_network(self):

		print "Building Network"
		
	def predict(self):

		print "Predicting Model"

	def fit(self):

		print "Training Model"