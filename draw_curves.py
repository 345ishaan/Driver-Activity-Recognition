import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from pdb import set_trace as brk

train_data = np.load('/home/ishan_shashank/state_farm/code/git_ishan/State_Farm/code/plot_train_vgg.npy')
val_data = np.load('/home/ishan_shashank/state_farm/code/git_ishan/State_Farm/code/plot_val_vgg.npy')
t = np.arange(train_data.shape[0])

tr = plt.plot(t,train_data,'r',label='Training') # plotting t,a separately 
val = plt.plot(t,val_data,'b',label='Validation') # plotting t,b separately 
# plt.legend([tr, val], ["Training", "Validation"])
plt.legend(loc='bottom right')
plt.xlabel('Number of iterations')
plt.ylabel('Accuracy')
plt.show()