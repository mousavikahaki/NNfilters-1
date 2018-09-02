import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import common_functions as cf

################# define functions
def weight_variable(shape):
    # From the mnist tutorial
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def AppCTCIm_valid(sess, Im, label, pad_x, pad_y, pad_z, out_x, out_y, out_z):    
    win_x = out_x + 2*pad_x
    win_y = out_y + 2*pad_y
    win_z = out_z + 2*pad_z
    output_ = np.zeros([Im.shape[0]-2*pad_x,Im.shape[1]-2*pad_y,Im.shape[2]-2*pad_z])
    input_batch = np.zeros([output_.size//(out_x*out_y*out_z),win_x*win_y*win_z])
    label_batch = np.zeros([output_.size//(out_x*out_y*out_z),out_x*out_y*out_z])
    ind = 0
    for i in range(0,Im.shape[0]-win_x,out_x):
        for j in range(0,Im.shape[1]-win_y,out_y):
            for k in range(0,Im.shape[2]-win_z,out_z):
                sub = Im[i:i+win_x,j:j+win_y,k:k+win_z]
                sub_label = label[i+pad_x:i+out_x+pad_x,j+pad_y:j+out_y+pad_y,k+pad_z:k+out_z+pad_z] 
                input_batch[ind,:] = sub.flatten('F')
                label_batch[ind,:] = sub_label.flatten('F') 
                ind = ind + 1
    feed = {x: input_batch, y_: label_batch, keep_prob: 1}
    temp_IM, loss_out, merged_ = sess.run([y,loss,merged], feed_dict=feed)
    return output_, loss_out, merged_

################### parameters
out_x = 8 # must be a power of 2
out_y = 8 # must be a power of 2
out_z = 4 # must be a power of 2
pad_x = 10 # padding in x
pad_y = 10 # padding in y
pad_z = 3 # padding in z
learning_rate = 0.1
keep_p = 1 # 1 - drop out probability
batch_size = 50 # batch size
max_train_steps = int(3e6) # maximum number of training steps
plotting_step = int(5e4) # update of the loss plot

################### paths
data_dir = '../../data/L1/'
train_IM_list = ['image_1.tif','image_2.tif','image_3.tif','image_6.tif']
train_label_list = ['label_1.tif','label_2.tif','label_3.tif','label_6.tif']
valid_IM_name = 'image_4.tif'
valid_label_name = 'label_4.tif'

model_dir = '../../data/training_result/'
folder_name = 'CTC_pad_1'
cf.createFolder(model_dir + folder_name)

###################
win_x = out_x + 2*pad_x
win_y = out_y + 2*pad_y
win_z = out_z + 2*pad_z

###################
IM_size_list = [None] * len(train_IM_list)
for i in range(len(train_IM_list)):
  phantom_IM, phantom_label = cf.get_data(data_dir+train_IM_list[i],data_dir+train_label_list[i])
  IM_size_list[i] = phantom_IM.size
N_total = sum(IM_size_list)
IM_ind = list(range(len(train_IM_list)))

################### creating network
# placeholders for the images
x = tf.placeholder(tf.float32, shape=[None, win_x*win_y*win_z], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, out_x*out_y*out_z], name="y_")

# placeholder for dropout
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
# first fully connected layer with 50 neurons using tanh activation
W1 = weight_variable([win_x*win_y*win_z, 100])
b1 = bias_variable([100])
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h1d = tf.nn.dropout(h1, keep_prob)
# second fully connected layer with 50 neurons using tanh activation
W2 = weight_variable([100, 50])
b2 = bias_variable([50])
l2 = tf.nn.relu(tf.matmul(h1d, W2) + b2)
l2d = tf.nn.dropout(l2, keep_prob)
# third fully connected layer with 50 neurons using tanh activation
W3 = weight_variable([50, 100])
b3 = bias_variable([100])
l3 = tf.nn.relu(tf.matmul(l2d, W3) + b3)
l3d = tf.nn.dropout(l3, keep_prob)
# readout layer
Wo = weight_variable([100, out_x*out_y*out_z])
bo = bias_variable([out_x*out_y*out_z])
final_in = tf.matmul(l3d, Wo) + bo
y = tf.nn.sigmoid(final_in, name="y")
loss = tf.reduce_mean(y_ * -tf.log(y) + (1 - y_) * -tf.log(1 - y))
merged = tf.summary.scalar('loss',loss)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

################### generating validation data
valid_IM, valid_label = cf.get_data(data_dir+valid_IM_name,data_dir+valid_label_name)      
valid_IM = valid_IM[370:498,210:338,21:37]
valid_label = valid_label[370:498,210:338,21:37]

px = plt.figure(1)
plt.imshow(cf.max_proj(valid_IM), cmap = 'gray')
px.show()
plt.pause(1)

py = plt.figure(2)
plt.imshow(cf.max_proj(valid_label), cmap = 'gray')
py.show()
plt.pause(1)

valid_label = np.pad(valid_label, ((pad_x,pad_x),(pad_y,pad_y),(pad_z,pad_z)), 'constant', constant_values=0)
valid_IM = np.pad(valid_IM, ((pad_x,pad_x),(pad_y,pad_y),(pad_z,pad_z)), 'constant', constant_values=0)

################### training
start_time = time.time()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep = (max_train_steps//plotting_step + 1))
sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter( model_dir+folder_name+'/train', sess.graph)
sess.run(tf.global_variables_initializer())  

ind = 0

dummy_train = np.zeros([batch_size, win_x*win_y*win_z])
dummy_label = np.zeros([batch_size, out_x*out_y*out_z])        

current_ind = -1
z = 0
s = 0
while s < max_train_steps:
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
    i = pixel_ind[m] // ((train_IM.shape[1] - win_y + 1)*(train_IM.shape[2] - win_z + 1)) % (train_IM.shape[0] - win_x + 1)
    j = ( pixel_ind[m] // (train_IM.shape[2] - win_z + 1) ) % (train_IM.shape[1] - win_y + 1)
    k = pixel_ind[m] % (train_IM.shape[2] - win_z + 1)
    temp_train = train_IM[i:i+win_x,j:j+win_y,k:k+win_z]
    temp_label = train_label[i+pad_x:i+out_x+pad_x,j+pad_y:j+out_y+pad_y,k+pad_z:k+out_z+pad_z]
    ind_trans = random.randrange(8)
    if ind_trans > 3:
      temp_train = np.flip(temp_train, 0)
      temp_label = np.flip(temp_label, 0)
    temp_train = np.rot90(temp_train, ind_trans % 4 )
    temp_label = np.rot90(temp_label, ind_trans % 4 )
    dummy_train[bi,:] = temp_train.flatten('F')
    dummy_label[bi,:] = temp_label.flatten('F')
    bi = bi + 1     
    z = z + 1
    
  feed = {x : dummy_train, y_: dummy_label, keep_prob: keep_p}  
  train_step.run(feed_dict=feed)
  s = s + 1 
  if s % plotting_step == 0:
    saver.save(sess, model_dir+folder_name+'/model'+str(s))
    train_out, train_loss, summary = AppCTCIm_valid(sess, valid_IM, valid_label, pad_x, pad_y, pad_z, out_x, out_y, out_z)
    train_writer.add_summary(summary,s)
    print("step %d, training loss: %g" % (s, train_loss))    
    print("--- %s seconds ---" % (time.time() - start_time))

########### Save session
saver.save(sess, model_dir+folder_name+'/model')
np.savez(model_dir+folder_name+'/var.npz',out_x = out_x,out_y=out_y,out_z=out_z,pad_x = pad_x,pad_y=pad_y,pad_z=pad_z,keep_p=keep_p,learning_rate=learning_rate, batch_size = batch_size, plotting_step = plotting_step, z = z, s = s)