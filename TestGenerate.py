# =============================================================================
# Modified version of https://github.com/ricedsp/D-AMP_Toolbox
# Takes images from a folder and calculates avg psnr, time, ssim and mse
# =============================================================================

import numpy as np
import argparse
import tensorflow as tf
import time
import LearnedDAMP as LDAMP
from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt
import h5py
from PIL import Image
from skimage import io
from psnr import psnr
from skimage.measure import compare_ssim as ssim
import os
import errno
import json

useSURE=False#Use the network trained with ground-truth data or with SURE
sigma_w=25./255.#Noise std, normalized 
data_set  ="NLM_XRAYS"



# some variables 
n_DnCNN_layers=16


input_dir = "./TestImages/"+data_set
# cleaned images will be saved here.
if useSURE:
    output_dir = "Results/"+data_set+"/SURE"
else:
    output_dir = "Results/"+data_set+"/MSE"   
    
test_image_names = os.listdir(input_dir)
# 
avg_psnr = 0
avg_mse = 0
avg_time =0
avg_ssim = 0



    
avg_results = {}
    
for image_name in test_image_names:
    test_images = Image.open(input_dir+"/"+image_name).convert('L')
    test_images = np.array(test_images).astype("float32")/255
    
    ## Network Parameters
    height_img = test_images.shape[0]
    width_img = test_images.shape[1]
    channel_img = 1 # RGB -> 3, Grayscale -> 1
    filter_height = 3
    filter_width = 3
    num_filters = 64
  
    ## Problem Parameters
    
    n=channel_img*height_img*width_img
    ## Training Parameters
    BATCH_SIZE = 1
    # Parameters to to initalize weights. Won't be used if old weights are loaded
    init_mu = 0
    init_sigma = 0.1
    
    train_start_time=time.time()
    
    ## Clear all the old variables, tensors, etc.
    tf.reset_default_graph()
    
    LDAMP.SetNetworkParams(new_height_img=height_img, new_width_img=width_img, new_channel_img=channel_img, \
                           new_filter_height=filter_height, new_filter_width=filter_width, new_num_filters=num_filters, \
                           new_n_DnCNN_layers=n_DnCNN_layers, new_n_DAMP_layers=None,
                           new_sampling_rate=None, \
                           new_BATCH_SIZE=BATCH_SIZE, new_sigma_w=sigma_w, new_n=n, new_m=None, new_training=False)
    LDAMP.ListNetworkParameters()
    
    # tf Graph input
    x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])
    
    ## Construct the measurement model and handles/placeholders
    y_measured = LDAMP.AddNoise(x_true,sigma_w)
    
    ## Initialize the variable theta which stores the weights and biases
    theta_dncnn=LDAMP.init_vars_DnCNN(init_mu, init_sigma)
    
    ## Construct the reconstruction model
    x_hat = LDAMP.DnCNN(y_measured,None,theta_dncnn,training=False)
    
    LDAMP.CountParameters()
    
    assert (len(test_images)>=BATCH_SIZE), "Requested too much Test data"
    
    x_test = test_images#np.array(test_images)#np.transpose( np.reshape(test_images[0:BATCH_SIZE], (BATCH_SIZE, height_img * width_img * channel_img)))
    x_test =x_test.reshape(-1,1)
    with tf.Session() as sess:
        y_test=sess.run(y_measured,feed_dict={x_true: x_test})
    
    ## Train the Model
    saver = tf.train.Saver()  # defaults to saving all variables
    saver_dict={}
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sigma_w_min=sigma_w*255.
        sigma_w_max=sigma_w*255.
    
        save_name = LDAMP.GenDnCNNFilename(sigma_w_min/255.,sigma_w_max/255.,useSURE=useSURE)
        save_name_chckpt = save_name + ".ckpt"
        saver.restore(sess, save_name_chckpt)
    
        print("Reconstructing Signal")
        start_time = time.time()
        [reconstructed_test_images]= sess.run([x_hat], feed_dict={y_measured:y_test})
        time_taken=time.time()-start_time
        result_folder = str(n_DnCNN_layers)+"L"+str(int(sigma_w_max))+"N"
        save_directory_noisy = output_dir+"/"+result_folder+"/Noisy/"
        save_directory_clean = output_dir+"/"+result_folder+"/Clean/"
        
        # create directory if did not exists already
        if not os.path.exists(os.path.dirname(save_directory_noisy)):
            try:
                os.makedirs(os.path.dirname(save_directory_noisy))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise    
        if not os.path.exists(os.path.dirname(save_directory_clean)):
            try:
                os.makedirs(os.path.dirname(save_directory_clean))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        plt.imsave(save_directory_noisy+"/"+image_name,y_test.reshape(height_img,width_img), cmap='gray')
        plt.imsave(save_directory_clean+"/"+image_name,reconstructed_test_images.reshape(height_img,width_img), cmap='gray')
        avg_psnr += psnr(reconstructed_test_images,x_test,1)
        avg_mse += np.linalg.norm(reconstructed_test_images-x_test)
        avg_time  += time_taken
        avg_ssim += ssim(reconstructed_test_images.reshape(height_img,width_img),x_test.reshape(height_img,width_img))
        
avg_results = {"Average PSNR":avg_psnr/len(test_image_names), "Average SSIM":avg_ssim/len(test_image_names),
                                                                                         "Average MSE":avg_mse/len(test_image_names), "Average Time Taken":avg_time/len(test_image_names)}        
result_dir = output_dir+"/"+result_folder+"/results.json"
with open(result_dir, 'w') as fp:
    json.dump(avg_results, fp)
print("average psnr:",avg_psnr/len(test_image_names),"average mse",avg_mse/len(test_image_names),
      "average time",avg_time/len(test_image_names), "average sso,",avg_ssim/len(test_image_names))