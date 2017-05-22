import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('subject'+'.mp4',fourcc, 5.0, (640,480), True)

N_CLASSES = 10
N_SUBJECTS = 26
TRAIN_PATH = '../state_farm_data/imgs/train/'
VISUALISE = False
GENERATE_DATA = True

tfrecords_filename = 'statefarm.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def make_tfrecord(data_frame):

	num_iterations = 0

	for row in tqdm(range(data_frame.shape[0])):
		try:
			img_raw = np.asarray(Image.open(TRAIN_PATH+str(data_frame[row,1])+'/'+str(data_frame[row,2])))
			w = img_raw.shape[1]
			h = img_raw.shape[0]
			if len(img_raw.shape) == 2 or img_raw.shape[2] == 1:
				print "Grayscale"
				continue
			label = int(data_frame[row,1].split('c')[1])

			img_raw = img_raw.tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
				'image_raw':_bytes_feature(img_raw),
				'width': _int64_feature(w),
				'height': _int64_feature(h),
				'class':  _int64_feature(label)
				}))
			
			
			writer.write(example.SerializeToString())
			num_iterations += 1

		except Exception as e:
			print e
	print "Written {} Images".format(num_iterations)
	writer.close()

def visualise(df):
	for i in range(df.shape[0]):
		if df[i,0] == 'p012' and df[i,1] == 'c8': 
			img = np.asarray(Image.open(TRAIN_PATH+str(df[i,1])+'/'+str(df[i,2])))
			img = cv2.resize(img,(640,480))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			print img.shape
			out.write((img).astype('u1'))
	out.release()


if __name__ == '__main__':
	df = pd.read_csv('../state_farm_data/driver_imgs_list.csv')
	df = np.asarray(df)
	if VISUALISE:
		visualise(df)
	if GENERATE_DATA:
		make_tfrecord(df)

	



