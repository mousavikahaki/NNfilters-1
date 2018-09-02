import numpy as np
import tensorflow as tf
import common_functions as cf

################# define functions
def AppCCIm_test(sess, Im, pad_x, pad_y, pad_z):
    output = np.zeros(Im.shape)
    Im = np.pad(Im, ((pad_x,pad_x),(pad_y,pad_y),(pad_z,pad_z)), 'constant', constant_values=0)
    win_x = 2*pad_x + 1
    win_y = 2*pad_y + 1
    win_z = 2*pad_z + 1
    output_batch = np.zeros([output.shape[0],win_x*win_y*win_z])
    for k in range(output.shape[2]):
        for i in range(output.shape[1]):
            for j in range(output.shape[0]):
              output_batch[j,:] = Im[i:i+win_x,j:j+win_y,k:k+win_z].reshape(1,win_x*win_y*win_z, order = 'F')
            temp = sess.run(y, feed_dict={x: output_batch, keep_prob: 1})
            output[i,:,k] = temp.reshape(output.shape[0], order = 'F')
    return output

################### paths
data_dir = '../../data/L1/'
test_IM_name = 'image_5.tif' 
test_label_name = 'label_5.tif' 

model_dir = '../../data/training_result/'
folder_name = 'CC_0'

############### loading parameters
model_var = np.load(model_dir+folder_name+'/var.npz')
pad_x = model_var['pad_x']
pad_y = model_var['pad_y']
pad_z = model_var['pad_z']

################### loading model
sess = tf.Session()
saver = tf.train.import_meta_graph(model_dir+folder_name+"/model.meta")
saver.restore(sess,model_dir+folder_name+"/model")
#####################
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("y_:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
y = graph.get_tensor_by_name("y:0")

################### loading test data
test_IM, test_label = cf.get_data(data_dir+test_IM_name,data_dir+test_label_name)

####################### testing    
test_output = AppCCIm_test(sess,test_IM, pad_x, pad_y, pad_z)

####################### showing test result
cf.plot_result(test_IM,test_output,test_label)




