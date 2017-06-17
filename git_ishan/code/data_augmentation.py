import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import shutil

data_frame = pd.read_csv('/home/ishan_shashank/state_farm/data/driver_imgs_list.csv')
data_frame = np.asarray(data_frame)

data_path = '/home/ishan_shashank/state_farm/data/imgs/train/'

if os.path.exists('/home/ishan_shashank/state_farm/split_data/training/'):
	os.makedirs('/home/ishan_shashank/state_farm/split_data/training/')
if os.path.exists('/home/ishan_shashank/state_farm/split_data/validation/'):
	os.makedirs('/home/ishan_shashank/state_farm/split_data/validation/')

subject_ids = map(lambda id : int(id.split('p')[1]),np.unique(data_frame[:,0]).tolist())
# val_ids = np.random.choice(subject_ids, N_CLASSES, False)
# val_class = np.random.choice(np.arange(N_CLASSES), N_CLASSES, False)
# val_dict ={}
# for id,cls in zip(val_ids, val_class):
# 	val_dict['p'+'0'*(3-len(str(id)))+str(id)] = cls
val_ids = np.random.choice(subject_ids, 3, False)
val_dict ={}
for id in val_ids:
	val_dict['p'+'0'*(3-len(str(id)))+str(id)] = 'all'

count_train = 0
count_val = 0


for row in tqdm(range(data_frame.shape[0])):
	try:
		#if val_dict.get(str(data_frame[row,0])) == label:

		if str(data_frame[row,0]) in val_dict:
			count_val += 1
			if os.path.exists('/home/ishan_shashank/state_farm/split_data/validation/'+str(data_frame[row,1])):
				shutil.copy(data_path+str(data_frame[row,1])+'/'+str(data_frame[row,2]), '/home/ishan_shashank/state_farm/split_data/validation/'+str(data_frame[row,1]))
			else:
				os.makedirs('/home/ishan_shashank/state_farm/split_data/validation/'+str(data_frame[row,1]))
				shutil.copy(data_path+str(data_frame[row,1])+'/'+str(data_frame[row,2]), '/home/ishan_shashank/state_farm/split_data/validation/'+str(data_frame[row,1]))
		else:
			count_train += 1
			if os.path.exists('/home/ishan_shashank/state_farm/split_data/training/'+str(data_frame[row,1])):
				shutil.copy(data_path+str(data_frame[row,1])+'/'+str(data_frame[row,2]), '/home/ishan_shashank/state_farm/split_data/training/'+str(data_frame[row,1]))
			else:
				os.makedirs('/home/ishan_shashank/state_farm/split_data/training/'+str(data_frame[row,1]))
				shutil.copy(data_path+str(data_frame[row,1])+'/'+str(data_frame[row,2]), '/home/ishan_shashank/state_farm/split_data/training/'+str(data_frame[row,1]))
		 
	except Exception as e:
		print e
	
print "Written {} in Val Images and {} in Train Images".format(count_val,count_train)
	
