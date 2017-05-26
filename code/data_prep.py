import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('subject'+'.mp4',fourcc, 5.0, (640,480), True)


# c0: safe driving
# c1: texting - right
# c2: talking on the phone - right
# c3: texting - left
# c4: talking on the phone - left
# c5: operating the radio
# c6: drinking
# c7: reaching behind
# c8: hair and makeup
# c9: talking to passenger
N_CLASSES = 10
N_SUBJECTS = 26
TRAIN_PATH = '/home/admin/Documents/state_farm/data/imgs/train/'
VISUALISE = False
GENERATE_DATA = True

tfrecords_train_filename = './statefarm_train.tfrecords'
writer_train = tf.python_io.TFRecordWriter(tfrecords_train_filename)
tfrecords_val_filename = './statefarm_val.tfrecords'
writer_val = tf.python_io.TFRecordWriter(tfrecords_val_filename)


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def make_tfrecord(data_frame,val_dict):

	num_val = 0
	num_train = 0

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
			
			#if val_dict.get(str(data_frame[row,0])) == label:
			if  str(data_frame[row,0]) in val_dict:
				writer_val.write(example.SerializeToString())
				num_val += 1
			else:
				writer_train.write(example.SerializeToString())
				num_train += 1

		except Exception as e:
			print e
	print "Written {} in Val Images and {} in Train Images".format(num_val,num_train)
	writer_train.close()
	writer_val.close()

def extract_tfrecord():
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
	save_data = None
	save_euler = []
	for string_record in record_iterator:
		example = tf.train.Example()
		example.ParseFromString(string_record)

		img_string = example.features.feature['image_raw'].bytes_list.value[0]
		img_width = int(example.features.feature['width'].int64_list.value[0])
		img_height = int(example.features.feature['height'].int64_list.value[0])
		labels = int(example.features.feature['class'].int64_list.value[0])
		img = np.fromstring(img_string, dtype=np.uint8).reshape(img_height,img_width,3)
		


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
	df = pd.read_csv('/home/admin/Documents/state_farm/data/driver_imgs_list.csv')
	df = np.asarray(df)
	
	subject_ids = map(lambda id : int(id.split('p')[1]),np.unique(df[:,0]).tolist())
	# val_ids = np.random.choice(subject_ids, N_CLASSES, False)
	# val_class = np.random.choice(np.arange(N_CLASSES), N_CLASSES, False)
	# val_dict ={}
	# for id,cls in zip(val_ids, val_class):
	# 	val_dict['p'+'0'*(3-len(str(id)))+str(id)] = cls
	val_ids = np.random.choice(subject_ids, 3, False)
	val_dict ={}
	for id in val_ids:
		val_dict['p'+'0'*(3-len(str(id)))+str(id)] = 'all'

	if VISUALISE:
		visualise(df)
	if GENERATE_DATA:
		make_tfrecord(df,val_dict)
		

	



