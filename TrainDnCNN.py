__author__ = 'cmetzler&alimousavi'

import numpy as np
import argparse
import tensorflow as tf
import time
import LearnedDAMP as LDAMP
from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
variables = [[10.0,16],[10.0,20],[75.0,16]]
for i in range(len(variables)):
    tf.reset_default_graph()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Use debugger to track down bad values during training")
    parser.add_argument(
        "--DnCNN_layers",
        type=int,
        default=variables[i][1],
        help="How many DnCNN layers per network")
    parser.add_argument(
        "--sigma_w_min",
        type=float,
        default=variables[i][0],
        help="Lowest noise level used to train network")
    parser.add_argument(
         "--sigma_w_max",
        type=float,
        default=variables[i][0],
        help="Highest noise level used to train network")
    parser.add_argument(
        "--loss_func",
        type=str,
        default="SURE",#Options are SURE or MSE
        help="Which loss function to use")
    FLAGS, unparsed = parser.parse_known_args()
    
    print((FLAGS))
    
    ## Network Parameters
    height_img = 40#50
    width_img = 40#50
    channel_img = 1 # RGB -> 3, Grayscale -> 1
    filter_height = 3
    filter_width = 3
    num_filters = 64
    n_DnCNN_layers= FLAGS.DnCNN_layers
    
    # summary counter 
    count = 0
    countVal = 0
    ## Training Parameters
    max_Epoch_Fails=5#How many training epochs to run without improvement in the validation error
    ResumeTraining=False #Load weights from a network you've already trained a little
    learning_rates = [0.001]#, 0.0001]#, 0.00001]
    EPOCHS = 5
    n_Train_Images=128*6170#3000
    n_Val_Images=128*73#10000#Must be less than 21504
    BATCH_SIZE = 64
    
    loss_func=FLAGS.loss_func
    if loss_func=='SURE':
        useSURE=True
    else:
        useSURE=False
    
    ## Problem Parameters
    sigma_w_min=FLAGS.sigma_w_min/255.#Noise std
    sigma_w_max=FLAGS.sigma_w_max/255.#Noise std
    n=channel_img*height_img*width_img
    
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
                           new_BATCH_SIZE=BATCH_SIZE, new_sigma_w=None, new_n=n, new_m=None, new_training=True)
    LDAMP.ListNetworkParameters()
    
    # tf Graph input
    training_tf = tf.placeholder(tf.bool, name='training')
    sigma_w_tf = tf.placeholder(tf.float32)
    x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])
    
    ## Construct the measurement model and handles/placeholders
    y_measured = LDAMP.AddNoise(x_true,sigma_w_tf)
    
    ## Initialize the variable theta which stores the weights and biases
    theta_dncnn=LDAMP.init_vars_DnCNN(init_mu, init_sigma)
    
    ## Construct the reconstruction model
    #x_hat = LDAMP.DnCNN(y_measured,None,theta_dncnn,training=training_tf)
    [x_hat, div_overN] = LDAMP.DnCNN_wrapper(y_measured,None,theta_dncnn,training=training_tf)
    
    ## Define loss and optimizer
    
    nfp=np.float32(height_img*width_img)
    if useSURE:
        cost = LDAMP.MCSURE_loss(x_hat,div_overN,y_measured,sigma_w_tf)
        summ = tf.summary.scalar('SURE_LOSS', cost)
    else:
        cost = tf.nn.l2_loss(x_true-x_hat)* 1./ nfp
        summ = tf.summary.scalar('MSE_LOSS', cost)
    LDAMP.CountParameters()
    merge = tf.summary.merge_all()
    ## Load and Preprocess Training Data
    #Training data was generated by GeneratingTrainingImages.m and ConvertImagestoNpyArrays.py
    train_images = (np.load('./TrainingData/TrainingData_patch'+str(height_img)+'.npy')/255).astype('float32')
    
    #train_images=train_images[range(n_Train_Images),0,:,:]
    #train_images = train_images[0:200,:,:,:]
    assert (len(train_images)>=n_Train_Images), "Requested too much training data"
    val_images = (np.load('./TrainingData/ValidationData_patch'+str(height_img)+'.npy')/255).astype('float32')
    #val_images = val_images[0:100,:,:,:]
    #val_images=val_images[:,0,:,:]
    assert (len(val_images)>=n_Val_Images), "Requested too much validation data"
    
    x_train = np.transpose(np.reshape(train_images, (-1, channel_img * height_img * width_img)))
    x_val = np.transpose(np.reshape(val_images, (-1, channel_img * height_img * width_img)))
    
    ## Train the Model
    for learning_rate in learning_rates:
        optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate) # Train all the variables
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step. A.llows us to update averages w/in BN
            optimizer = optimizer0.minimize(cost)
    
    
        saver_best = tf.train.Saver()  # defaults to saving all variables
        saver_dict={}
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            train_writer = tf.summary.FileWriter( './logs/ '+str(FLAGS.sigma_w_min)+"layers"+str(FLAGS.DnCNN_layers ), sess.graph)
            sess.run(tf.global_variables_initializer())#Seems to be necessary for the batch normalization layers for some reason.
    
            # if FLAGS.debug:
            #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            #     sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    
            start_time = time.time()
            print(("Load Initial Weights ..."))
            if ResumeTraining or learning_rate!=learning_rates[0]:
                ##Load previous values for the weights and BNs
                saver_initvars_name_chckpt=LDAMP.GenDnCNNFilename(sigma_w_min,sigma_w_max,useSURE=useSURE)+ ".ckpt"
                for l in range(0, n_DnCNN_layers):
                    saver_dict.update({"l" + str(l) + "/w": theta_dncnn[0][l]})
                for l in range(1,n_DnCNN_layers-1):#Associate variance, means, and beta
                    gamma_name = "l" + str(l) + "/BN/gamma:0"
                    beta_name="l" + str(l) + "/BN/beta:0"
                    var_name="l" + str(l) + "/BN/moving_variance:0"
                    mean_name="l" + str(l) + "/BN/moving_mean:0"
                    gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                    beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                    moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                    moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                    saver_dict.update({"l" + str(l) + "/BN/gamma": gamma})
                    saver_dict.update({"l" + str(l) + "/BN/beta": beta})
                    saver_dict.update({"l" + str(l) + "/BN/moving_variance": moving_variance})
                    saver_dict.update({"l" + str(l) + "/BN/moving_mean": moving_mean})
                saver_initvars = tf.train.Saver(saver_dict)
                saver_initvars.restore(sess, saver_initvars_name_chckpt)
                # saver_initvars = tf.train.Saver()
                # saver_initvars.restore(sess, saver_initvars_name_chckpt)
            else:
                pass
            time_taken = time.time() - start_time
    
            print("Training ...")
            print()
            if __name__ == '__main__':
                save_name = LDAMP.GenDnCNNFilename(sigma_w_min,sigma_w_max,useSURE=useSURE)
                save_name_chckpt = save_name + ".ckpt"
                val_values = []
                print("Initial Weights Validation Value:")
               # rand_inds = np.random.choice(len(val_images), n_Val_Images, replace=False)
                start_time = time.time()
                for offset in range(0,10):#n_Val_Images - BATCH_SIZE + 1, BATCH_SIZE-1):  # Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                    end = offset + BATCH_SIZE
    
                    batch_x_val = x_val[:, offset:end]
                    sigma_w_thisBatch = sigma_w_min + np.random.rand() * (sigma_w_max - sigma_w_min)
    
                    # Run optimization.
                    loss_val = sess.run(cost, feed_dict={x_true: batch_x_val, sigma_w_tf: sigma_w_thisBatch, training_tf:False})
                    val_values.append(loss_val)
    
                time_taken = time.time() - start_time
                print( np.mean(val_values))
                best_val_error = np.mean(val_values)
                best_sess = sess
                print( "********************")
                save_path = saver_best.save(best_sess, save_name_chckpt)
                print(("Initial session model saved in file: %s" % save_path))
                failed_epochs=0
                for i in range(EPOCHS):
                    if failed_epochs>=max_Epoch_Fails:
                        break
                    train_values = []
                    print( ("This Training iteration ..."))
                    #rand_inds=np.random.choice(len(train_images), n_Train_Images,replace=False)
                    start_time = time.time()
                    for offset in tqdm(range(0,n_Train_Images-BATCH_SIZE+1, BATCH_SIZE)):#Subtract batch size-1 to avoid errors when len(train_images) is not a multiple of the batch size
                        end = offset + BATCH_SIZE
    
                        batch_x_train = x_train[:, offset:end]
                        sigma_w_thisBatch = sigma_w_min+np.random.rand()*(sigma_w_max-sigma_w_min)
    
                        # Run optimization.
                        _, loss_val,cost_summ = sess.run([optimizer,cost,merge], feed_dict={x_true: batch_x_train, sigma_w_tf: sigma_w_thisBatch, training_tf:True})#Feed dict names should match with the placeholders
                        train_values.append(loss_val)
                        train_writer.add_summary(cost_summ, count)
                        count +=1
                    time_taken = time.time() - start_time
                    print( np.mean(train_values))
                    val_values = []
                    print(("EPOCH ",i+1," Validation Value:" ))
                    #rand_inds = np.random.choice(len(val_images), n_Val_Images, replace=False)
                    start_time = time.time()
                    for offset in range(0,n_Val_Images-BATCH_SIZE+1, BATCH_SIZE):#Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                        end = offset + BATCH_SIZE
    
                        batch_x_val = x_val[:, offset:end]
                        sigma_w_thisBatch = sigma_w_min + np.random.rand() * (sigma_w_max - sigma_w_min)
    
                        # Run optimization.
                        loss_val = sess.run(cost, feed_dict={x_true: batch_x_val, sigma_w_tf: sigma_w_thisBatch, training_tf:False})
                        val_values.append(loss_val)
                        summary=tf.Summary()
                        summary.value.add(tag='validation_loss', simple_value = loss_val)
                        train_writer.add_summary(summary, countVal)
                    time_taken = time.time() - start_time
                    print(np.mean(val_values))
                    if(np.mean(val_values) < best_val_error):
                        failed_epochs=0
                        best_val_error = np.mean(val_values)
                        best_sess = sess
                        print( "********************")
                        save_path = saver_best.save(best_sess, save_name_chckpt)
                        print(("Best session model saved in file: %s" % save_path))
                    else:
                        failed_epochs=failed_epochs+1
                    print("********************")
    
    total_train_time=time.time()-train_start_time
    save_name_time = save_name + "_time.txt" 
    print(total_train_time)
    #f= open(save_name, 'wb')
    #f.write("Total Training Time ="+str(total_train_time))
    #f.close()
