import numpy as np
from keras.models import load_model
import common_functions as cf


################# define functions
def App3DUNetIm_test(model, Im, win_x, win_y, win_z, out_x, out_y, out_z):    
    stride_x = out_x
    stride_y = out_y
    stride_z = out_z
    output = np.zeros(Im.shape)

    for i in range(0,Im.shape[0],stride_x):
        for j in range(0,Im.shape[1],stride_y):
            for k in range(0,Im.shape[2]-stride_z,stride_z):
                sub = Im[i:i+stride_x,j:j+stride_y,k:k+stride_z]
                sub = np.expand_dims(np.expand_dims(sub, axis=0), axis=4)
                SM = output[i:i+out_x,j:j+out_y,k:k+out_z]
                temp_IM = model.predict(sub)
                temp_IM = np.squeeze(temp_IM)
                output[i:i+out_x,j:j+out_y,k:k+out_z] = np.maximum(SM,temp_IM)
    # dealing with boundary
    for i in range(0,Im.shape[0],stride_x):
        for j in range(0,Im.shape[1],stride_y):
            k = Im.shape[2]-stride_z
            sub = Im[i:i+stride_x,j:j+stride_y,k:k+stride_z]
            sub = np.expand_dims(np.expand_dims(sub, axis=0), axis=4)
            SM = output[i:i+out_x,j:j+out_y,k:k+out_z]
            temp_IM = model.predict(sub)
            temp_IM = np.squeeze(temp_IM)
            output[i:i+out_x,j:j+out_y,k:k+out_z] = np.maximum(SM,temp_IM)
    return output  

################### paths
data_dir = '../../data/L1/'
test_IM_name = 'image_5.tif' 
test_label_name = 'label_5.tif' 

model_dir = '../../data/training_result/'
folder_name = '3DUNet_0'

################### loading parameters
data = np.load(model_dir+folder_name+'/var.npz')
win_x = data['win_x']
win_y = data['win_y']
win_z = data['win_z']
out_x = win_x
out_y = win_y
out_z = win_z

########### Load model
model = load_model(model_dir+folder_name+'/model')

################### loading test data             
test_IM, test_label = cf.get_data(data_dir+test_IM_name,data_dir+test_label_name)

####################### testing 
test_output = App3DUNetIm_test(model, test_IM, win_x, win_y, win_z, out_x, out_y, out_z)

####################### showing test result
cf.plot_result(test_IM,test_output,test_label)
