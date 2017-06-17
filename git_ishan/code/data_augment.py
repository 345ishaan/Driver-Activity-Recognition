import os
import pandas as pd
import numpy as np
import cv2
from skimage import color
from skimage.transform import resize
from PIL import Image
from tqdm import tqdm
from pdb import set_trace as brk
import scipy.misc
import math


data_path = '/home/ishan_shashank/state_farm/split_data/training/'
new_data_path = '/home/ishan_shashank/state_farm/data_augmentation/new_data/'

dirs = map(lambda x : data_path+x,os.listdir(data_path))

if not os.path.exists(new_data_path):
	os.makedirs(new_data_path)

sv_percentage = np.arange(0.25,1.75,0.25)
h_percentage =  np.arange(0.25,1.75,0.25)

CROPPED_DIM = 227

def create_data():

	for dr in dirs:
		class_val = dr.split('/')[-1]

		if not os.path.exists(new_data_path+class_val):
			os.makedirs(new_data_path+class_val)
		ip_imgs = os.listdir(dr)
		for ip_img in ip_imgs:
			img = np.asarray(Image.open(dr+'/'+ip_img))
			rgb_to_hsv = color.rgb2hsv(img)
			count = 0
			for h_change in h_percentage:
				for sv_change in sv_percentage: 
					new_rgb_to_hsv = np.zeros_like(rgb_to_hsv)
					new_rgb_to_hsv[:,:,0] = (((rgb_to_hsv[:,:,0] - np.mean(rgb_to_hsv[:,:,0]) )/ np.std(rgb_to_hsv[:,:,0]))*(np.std(rgb_to_hsv[:,:,0]))) + (h_change* np.mean(rgb_to_hsv[:,:,0]))
					new_rgb_to_hsv[:,:,1] = (((rgb_to_hsv[:,:,1] - np.mean(rgb_to_hsv[:,:,1]) )/ np.std(rgb_to_hsv[:,:,1]))*(np.std(rgb_to_hsv[:,:,1]))) + (sv_change*np.mean(rgb_to_hsv[:,:,1]))
					new_rgb_to_hsv[:,:,2] = (((rgb_to_hsv[:,:,2] - np.mean(rgb_to_hsv[:,:,2]) )/ np.std(rgb_to_hsv[:,:,2]))*(np.std(rgb_to_hsv[:,:,2])))+ (sv_change*np.mean(rgb_to_hsv[:,:,2]))
					hsv_to_rgb = color.hsv2rgb(new_rgb_to_hsv)
					scipy.misc.imsave(os.path.join(new_data_path+class_val,ip_img.split('.')[0]+'_'+str(count)+".jpg"), hsv_to_rgb)
					#hsv_to_rgb.save(os.path.join(new_data_path+class_val,ip_img.split('.')[0]+'_'+str(count)+".jpg"))
					count += 1
			break
		break


def make_random_crops():

	for dr in dirs:
		class_val = dr.split('/')[-1]

		if not os.path.exists(new_data_path+class_val):
			os.makedirs(new_data_path+class_val)
		ip_imgs = os.listdir(dr)
		for ip_img in ip_imgs:
			img = np.asarray(Image.open(dr+'/'+ip_img))
			resize_256 = resize(img, (256, 256), mode='reflect')
			indices = [1,256-CROPPED_DIM+1] 
			count = 0
			for m in indices:
				for n in indices:
					new_img = resize_256[m:m+CROPPED_DIM-1,n:n+CROPPED_DIM-1,:]
					scipy.misc.imsave(os.path.join(new_data_path+class_val,ip_img.split('.')[0]+'_'+str(count)+".jpg"), new_img*255.0)
					count += 1
			center = (indices[1]/2)+1

			new_img = resize_256[center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:]

			scipy.misc.imsave(os.path.join(new_data_path+class_val,ip_img.split('.')[0]+'_'+str(count)+".jpg"), new_img*255.0)
			resize_256 = resize(img, (CROPPED_DIM, CROPPED_DIM), mode='reflect')
			scipy.misc.imsave(os.path.join(new_data_path+class_val,ip_img.split('.')[0]+".jpg"), resize_256*255.0)

			

if __name__ == '__main__':
	make_random_crops()









