import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('subject'+'.mp4',fourcc, 5.0, (640,480), True)

N_CLASSES = 10
N_SUBJECTS = 26
TRAIN_PATH = '../state_farm_data/imgs/train/'

if __name__ == '__main__':
	df = pd.read_csv('../state_farm_data/driver_imgs_list.csv')
	df = np.asarray(df)

	for i in range(df.shape[0]):
		if df[i,0] == 'p012' and df[i,1] == 'c8': 
			img = np.asarray(Image.open(TRAIN_PATH+str(df[i,1])+'/'+str(df[i,2])))
			img = cv2.resize(img,(640,480))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			print img.shape
			out.write((img).astype('u1'))
	out.release()

