import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv

fp = open('cf_matrix.txt','rb')
fp_w = open('vgg_cam.csv','wb')
fp_w_csv = csv.writer(fp_w)

labels_list = ['c0: safe driving','c1: texting - right','c2: talking on the phone - right','c3: texting - left','c4: talking on the phone - left','c5: operating the radio',
'c6: drinking',
'c7: reaching behind',
'c8: hair and makeup',
'c9: talking to passenger',
]
labels_list=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
fp_w_csv.writerow(labels_list)
counter = 0
for row in fp:
	list_ = row.split()
	if counter > 0:
		list_.append(labels_list[counter-1])
	list_.append(labels_list[counter])
	fp_w_csv.writerow(list_)
	print counter
	counter+=1
fp.close()
fp_w.close()
