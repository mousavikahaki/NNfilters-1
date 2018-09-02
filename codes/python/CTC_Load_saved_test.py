import numpy as np
import tensorflow as tf
import common_functions as cf

################# define functions
def AppCTCIm_test(sess, Im, pad_x, pad_y, pad_z, out_x, out_y, out_z):    
    output = np.zeros(Im.shape)
    Im = np.pad(Im, ((pad_x,pad_x),(pad_y,pad_y),(pad_z,pad_z)), 'constant', constant_values=0)
    win_x = out_x + 2*pad_x
    win_y = out_y + 2*pad_y
    win_z = out_z + 2*pad_z

    for i in range(0,Im.shape[0]-win_x+1,out_x):
        for j in range(0,Im.shape[1]-win_y+1,out_y):
            for k in range(0,Im.shape[2]-win_z+1,out_z):
                sub = Im[i:i+win_x,j:j+win_y,k:k+win_z].reshape(-1, win_x*win_y*win_z, order = 'F')
                feed = {x: sub, keep_prob: 1}
                SM = output[i:i+out_x,j:j+out_y,k:k+out_z]
                temp_IM = sess.run(y, feed_dict=feed)
                output[i:i+out_x,j:j+out_y,k:k+out_z] = np.maximum(SM,temp_IM.reshape(out_x, out_y, out_z, order = 'F'))
        for i in range(0,Im.shape[0]-win_x+1,out_x):
            for j in range(0,Im.shape[1]-win_y+1,out_y):
                k = Im.shape[2]-win_z
                sub = Im[i:i+win_x,j:j+win_y,k:k+win_z].reshape(-1, win_x*win_y*win_z, order = 'F')
                feed = {x: sub, keep_prob: 1}
                SM = output[i:i+out_x,j:j+out_y,k:k+out_z]
                temp_IM = sess.run(y, feed_dict=feed)
                output[i:i+out_x,j:j+out_y,k:k+out_z] = np.maximum(SM,temp_IM.reshape(out_x, out_y, out_z, order = 'F'))
    return output

################### paths
data_dir = '../../data/L1/'
test_IM_name = 'image_5.tif' 
test_label_name = 'label_5.tif' 

model_dir = '../../data/training_result/'
folder_name = 'CTC_0'

################### loading parameters
data = np.load(model_dir+folder_name+'/var.npz')
out_x = data['out_x']
out_y = data['out_y']
out_z = data['out_z']
pad_x = data['pad_x']
pad_y = data['pad_y']
pad_z = data['pad_z']

################### loading model
sess = tf.Session()
saver = tf.train.import_meta_graph(model_dir+folder_name+"/model.meta")
saver.restore(sess,model_dir+folder_name+"/model")
###################
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("y_:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
y = graph.get_tensor_by_name("y:0")

################### loading test data
test_IM, test_label = cf.get_data(data_dir+test_IM_name,data_dir+test_label_name)

################### testing
test_output = AppCTCIm_test(sess, test_IM, pad_x, pad_y, pad_z, out_x, out_y, out_z)

####################### showing test result
cf.plot_result(test_IM,test_output,test_label)

