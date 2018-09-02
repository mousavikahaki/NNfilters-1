# Adapted from https://github.com/zhixuhao/unet

import time
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Concatenate, Conv3D, Dropout, Input,
                          MaxPooling3D, UpSampling3D)
from keras.models import Model
from keras.optimizers import Adam
import random
import common_functions as cf


################# define functions
def Im2Data(Im,label,sub_size):
    L = Im.shape[0]//sub_size[0] * Im.shape[1]//sub_size[1] * Im.shape[2]//sub_size[2]
    x_val = np.zeros([L,sub_size[0],sub_size[1],sub_size[2],1])
    y_val = np.zeros([L,sub_size[0],sub_size[1],sub_size[2],1])
    l = 0
    for i in range(0,Im.shape[0]-sub_size[0]+1,sub_size[0]):
        for j in range(0,Im.shape[1]-sub_size[1]+1,sub_size[1]):
            for k in range(0,Im.shape[2]-sub_size[2]+1,sub_size[2]):
                x_val[l,:,:,:,0] = Im[i:i+sub_size[0],j:j+sub_size[1],k:k+sub_size[2]]
                y_val[l,:,:,:,0] = label[i:i+sub_size[0],j:j+sub_size[1],k:k+sub_size[2]]
                l = l + 1
    return x_val, y_val
    

################### parameters
win_x = 32 # x size of input and output, must be a power of 2
win_y = 32 # y size of input and output, must be a power of 2
win_z = 8 # z size of input and output, positive integer
learning_rate = 1
keep_p = 0.8 # 1 - drop out probability
batch_size = 50 # batch size
max_train_steps = int(2e4) # maximum number of training steps
plotting_step = int(1e3) # update of the loss plot 

################### paths
data_dir = '../../data/L1/'
train_IM_list = ['image_1.tif','image_2.tif','image_3.tif','image_6.tif']
train_label_list = ['label_1.tif','label_2.tif','label_3.tif','label_6.tif']
valid_IM_name = 'image_4.tif'
valid_label_name = 'label_4.tif'

model_dir = '../../data/training_result/'
folder_name = '3DUNet_1'
cf.createFolder(model_dir + folder_name)
################### 

pad_x = int(win_x/2)
pad_y = int(win_y/2)
pad_z = int(win_z/2)
out_x = win_x
out_y = win_y
out_z = win_z

###################
IM_size_list = [None] * len(train_IM_list)
for i in range(len(train_IM_list)):
  phantom_IM, phantom_label = cf.get_data(data_dir+train_IM_list[i],data_dir+train_label_list[i])
  IM_size_list[i] = phantom_IM.size
N_total = sum(IM_size_list)
IM_ind = list(range(len(train_IM_list)))

# -----------------------------------------------------------------------------
# Define model 
conv_properties = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}

input_im = Input(shape=(win_x, win_y, win_z, 1), name='input_im')

conv1 = Conv3D(8, (3, 3, 3), **conv_properties)(input_im)
pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

conv2 = Conv3D(8, (3, 3, 3), **conv_properties)(pool1)
pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

conv3 = Conv3D(4, (3, 3, 1), **conv_properties)(pool2)
pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

conv4 = Conv3D(4, (3, 3, 1), **conv_properties)(pool3)
drop4 = Dropout(0.2)(conv4)
pool4 = MaxPooling3D(pool_size=(2, 2, 1))(drop4)

conv5 = Conv3D(4, (3, 3, 1), **conv_properties)(pool4)
drop5 = Dropout(0.2)(conv5)

up6 = UpSampling3D(size=(2, 2, 1))(drop5)
up6 = Conv3D(4, (2, 2, 1), **conv_properties)(up6)
cat6 = Concatenate(axis=-1)([drop4, up6])
conv6 = Conv3D(4, (3, 3, 1), **conv_properties)(cat6)

up7 = UpSampling3D(size=(2, 2, 1))(conv6)
up7 = Conv3D(4, (2, 2, 1), **conv_properties)(up7)
cat7 = Concatenate(axis=-1)([conv3, up7])
conv7 = Conv3D(4, (3, 3, 1), **conv_properties)(cat7)

up8 = UpSampling3D(size=(2, 2, 1))(conv7)
up8 = Conv3D(8, (2, 2, 1), **conv_properties)(up8)
cat8 = Concatenate(axis=-1)([conv2, up8])  
conv8 = Conv3D(8, 3, **conv_properties)(cat8)

up9 = UpSampling3D(size=(2, 2, 1))(conv8)
up9 = Conv3D(8, (2, 2, 1), **conv_properties)(up9)
cat9 = Concatenate(axis=-1)([conv1, up9])
conv9 = Conv3D(8, (3, 3, 3), **conv_properties)(cat9)

output_im = Conv3D(1, (1, 1, 1), activation='sigmoid', name='output_im')(conv9)

model = Model(inputs=[input_im], outputs=[output_im])

model.compile(optimizer=Adam(), loss= 'binary_crossentropy', metrics=['accuracy'])

################### generating validation data
valid_IM, valid_label = cf.get_data(data_dir+valid_IM_name,data_dir+valid_label_name)
valid_IM = valid_IM[370:498,210:338,21:37]
valid_label = valid_label[370:498,210:338,21:37]

px = plt.figure(1)
plt.imshow(cf.max_proj(valid_IM), cmap = 'gray')
px.show()
plt.close()
plt.pause(1)

py = plt.figure(2)
plt.imshow(cf.max_proj(valid_label), cmap = 'gray')
py.show()
plt.close()
plt.pause(1)

x_val, y_val = Im2Data(valid_IM,valid_label,[win_x,win_y,win_z])

################### training 
start_time = time.time()            
training_steps = []
current_ind = -1
z = 0
s = 0
p_loss = plt.figure(3)
plt.title('loss')
while s < max_train_steps:
  dummy_train = np.zeros([batch_size, win_x, win_y, win_z, 1])
  dummy_label = np.zeros([batch_size, win_x, win_y, win_z, 1])
  bi = 0
  while bi < batch_size:
    m = z % N_total
    ind = 0
    while m >= IM_size_list[IM_ind[ind]]:
        m = m - IM_size_list[IM_ind[ind]]
        ind = ind + 1
    if ind != current_ind:
        train_IM, train_label = cf.get_data(data_dir+train_IM_list[IM_ind[ind]],data_dir+train_label_list[IM_ind[ind]])
        train_IM = np.pad(train_IM, ((pad_x,pad_x),(pad_y,pad_y),(pad_z,pad_z)), 'constant', constant_values=0)
        train_label = np.pad(train_label, ((pad_x,pad_x),(pad_y,pad_y),(pad_z,pad_z)), 'constant', constant_values=0)
        current_ind = ind
        pixel_ind = np.arange((train_IM.shape[0] - 2*pad_x)*(train_IM.shape[1] - 2*pad_y)*(train_IM.shape[2] - 2*pad_z))
        random.shuffle(pixel_ind)
    i = pixel_ind[m] // ((train_IM.shape[1] - 2*pad_y)*(train_IM.shape[2] - 2*pad_z))
    j = ( pixel_ind[m] // (train_IM.shape[2] - 2*pad_z) ) % (train_IM.shape[1] - 2*pad_y)
    k = pixel_ind[m] % (train_IM.shape[2] - 2*pad_z)
    temp_train = train_IM[i:i+win_x,j:j+win_y,k:k+win_z]
    temp_label = train_label[i:i+win_x,j:j+win_y,k:k+win_z]
    ind_trans = random.randrange(8)
    if ind_trans > 3:
        temp_train = np.flip(temp_train, 0)
        temp_label = np.flip(temp_label, 0)
    temp_train = np.rot90(temp_train, ind_trans % 4 )
    temp_label = np.rot90(temp_label, ind_trans % 4 )
    dummy_train[bi,:,:,:,0] = temp_train
    dummy_label[bi,:,:,:,0] = temp_label
    bi = bi + 1     
    z = z + 1

  history = model.fit(x=dummy_train, y=dummy_label, batch_size=batch_size, epochs=1, verbose=0, callbacks = None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
  s = s + 1
  
  if s % plotting_step == 0:
    model.save('../../data/training_result/'+folder_name+'/model'+str(s))
    valid_loss_acc = model.evaluate(x=x_val,y=y_val)
    vl, = plt.plot(s,valid_loss_acc[0], 'go')
    hl, = plt.plot(s,history.history['loss'], 'ro')
    plt.legend([vl,hl],['validation loss','batch loss'])
    p_loss.show()
    plt.pause(1)
    
    print("--- %s seconds ---" % (time.time() - start_time))

print("training %s seconds ---" % (time.time() - start_time))
  
########### Save session
model.save(model_dir+folder_name+'/model')
np.savez(model_dir+folder_name+'/var.npz', win_x = win_x, win_y=win_y, win_z=win_z, learning_rate=learning_rate, batch_size = batch_size, keep_p = keep_p, plotting_step = plotting_step, z = z, s = s)
