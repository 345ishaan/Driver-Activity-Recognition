import os
import sys
import argparse
import cv2
import numpy as np
import h5py
import pandas as pd
from ipdb import set_trace as brk
from tqdm import tqdm

class Datagen(object):

	def __init__(self,args):

		self.max_rnn_length = args.max_time_steps
		self.fps = args.fps
		self.randomize = True
		self.batch_size = args.batch_size
		self.stride = args.stride
		self.data_path = '/home/admin/Documents/action_recongition/data/five-video-classification-methods/data/'
		self.dataset = args.dataset
		self.cnn_spatial_size = args.spatial_feature_dim
		self.cnn_spatial_depth = args.spatial_feature_depth

		self.skip_frames  = int(30)/self.fps

		self.class_map = {} #Class Name to Label Mapping
		self.num_frames = [] #Number of frames per video sequence
		self.labels = [] #Labels to each video sequence

		if self.dataset == 'ucf11':
			self.class_map,self.num_frames,self.labels,self.total_frames = self.get_labels(self.data_path)
		
		assert len(self.num_frames) == len(self.labels)

		self.num_train_samples = len(self.labels)

		self.frame_seqs=[]
		self.label_seqs=[]
		self.video_length_seqs=[]

		start =0
		
		for v,n in enumerate(self.num_frames):
			end = start + n - self.max_rnn_length*self.skip_frames +1
			if start > end:
				end = start +1
			self.frame_seqs.extend(range(start,end,self.stride))

			for it in (range(start,end,self.stride)):
				self.label_seqs.append(self.labels[v])
				self.video_length_seqs.append(self.num_frames[v])
			start += n
		
		self.dataset_size = len(self.frame_seqs)

		print "DataSet Size {}".format(self.dataset_size)

		self.frame_seqs=np.asarray(self.frame_seqs)
		self.label_seqs=np.asarray(self.label_seqs)
		self.video_length_seqs=np.asarray(self.video_length_seqs)
		
		assert len(self.frame_seqs) == len(self.label_seqs)

		self.video_boundary = np.asarray(self.num_frames).cumsum()

		self.shuffle()

		self.pick = 0
		self.databag = None

	def shuffle(self):

		np.random.seed(seed=7)
		np.random.shuffle(self.frame_seqs)
		np.random.seed(seed=7)
		np.random.shuffle(self.label_seqs)
		np.random.seed(seed=7)
		np.random.shuffle(self.video_length_seqs)


	def get_labels(self,path):
		class_map={}
		labels=[]
		num_frames=[]
		df = pd.read_csv(path+'data_file.csv')
		df = np.asarray(df)
		
		counter = 0

		for i in xrange(df.shape[0]):
			if df[i,0] == 'train':
				num_frames.append(int(df[i,3]))
				if df[i,1] not in class_map:
					labels.append(class_map.get(df[i,1],0)+counter)
					counter += 1
				else:
					labels.append(class_map.get(df[i,1],0))

		total_frames  = np.sum(df[:,3].astype(np.int32))

		return class_map,num_frames,labels,total_frames

	def get_data(self):

		self.batch_data = np.zeros((self.batch_size,self.max_rnn_length,self.cnn_spatial_size,self.cnn_spatial_depth))
		self.batch_labels = np.zeros((self.max_rnn_length,self.batch_size))

		for i in xrange(self.batch_size):
			start = self.pick
			if self.video_length_seqs[start] < self.max_rnn_length*self.skip_frames:
				end = self.video_length_seqs[start]
				n = 1+ (int(end)/self.skip_frames)
				self.batch_data[i,:n,:,:] = self.databag[range(start,end,self.skip_frames),:,:]
				self.batch_data[i,n:,:,:] = np.tile(self.databag[n-1,:,:],[n-self.max_rnn_length,1,1])
			else:
				end = start + self.max_rnn_length*self.skip_frames
				self.batch_data[i,:,:,:] = self.databag[range(start,end,self.skip_frames),:,:]
			self.batch_labels[:,i] = np.tile(self.label_seqs[start],[self.max_rnn_length])


	def create_hdf5(self):
		
		f = h5py.File('all_images.hdf5','w')
		
		df = np.asarray(pd.read_csv(self.data_path+'data_file.csv'))
		df = df[np.where(df[:,0]=='train')]
		total_frames  = np.sum(df[:,3].astype(np.int32))

		image_set = f.create_dataset('images',(total_frames,240,320,3),dtype='uint8',shuffle=False)

		count  = 0 
		for i in tqdm(xrange(df.shape[0])):
			for j in xrange(int(df[i,3])):
				path = self.data_path+'train/'+str(df[i,1])+'/'+str(df[i,2])+'-'+'0'*(4-len(str(j+1)))+str(j+1)+'.jpg'
				
				image = cv2.imread(path).astype(np.uint8)
				image_set[count,:,:,:] = image
				count += 1
			
		print "here"

	def read_hdf5(self):

		data = h5py.File('all_images.hdf5','r')['images']
		brk()











