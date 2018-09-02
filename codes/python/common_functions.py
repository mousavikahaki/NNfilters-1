"""
Created on Mon Aug 20 14:29:06 2018

@author: Shih-Luen Wang
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os

def get_data(IM_path,label_path):

  IM = io.imread(IM_path).astype(float)
  IM = np.einsum('kij->ijk',IM)
  IM = (IM/255)
  label = io.imread(label_path).astype(float)
  label = np.einsum('kij->ijk',label)
  label = ((label==255)*0.5+(label != 0)*0.5)
  return IM, label

def max_proj(x):
  y = np.zeros((len(x),len(x[0])))
  for i in range(len(x)):
    for j in range(len(x[0])):
      y[i,j] = np.amax(x[i,j,:])
  return y

def plot_result(IM,output,label):
    p_I = plt.figure(1)
    plt.imshow(max_proj(IM), cmap = 'gray')
    p_I.show()

    p_O = plt.figure(2)
    plt.imshow(max_proj(output), cmap = 'gray')
    p_O.show()

    p_L = plt.figure(3)
    plt.imshow(max_proj(label), cmap = 'gray')
    p_L.show()
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)