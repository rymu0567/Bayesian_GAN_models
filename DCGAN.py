import numpy as np 
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import progressbar
import time
import shutil
import random


#############################################################################################################################################################################################################################
# %% Deleating File

directory = '/home/ryan/Documents/Python/APPM 5720 Bayes Stats/Project/code/Bayesian_GAN_models/'
shutil.rmtree(directory+'logs', ignore_errors=True)

#############################################################################################################################################################################################################################
# %% Class 

class Model(object):
    def __init__(self,):

        ###############################
        #Properties
        self.lr = 10**-4*5
        self.Noise_size = 100
        self.epoch_size = 10**2
        self.batch_size = 10**3
        self.keep_prob = .7
        
        ###############################
        #Dataset
        self.dataset_name = 'mnist'
        self.load_datasets()
        self.Noise_data = np.random.uniform(size = [self.X_data.shape[0],self.Noise_size])
        self.batch_len = self.X_data.shape[0]//self.batch_size
        self.X_width = self.X_height = self.X_data.shape[1]
        self.X_debth = self.X_data.shape[-1]
        self.latent_input_size = self.latent_data.shape[-1]
        self.num = self.latent_data.max()
        self.convert_to_tuples()
        
        ###############################
        #Other Properties        
        self.i = 0
        self.batch_bar = progressbar.ProgressBar()#maxval = self.batch_size-1)
        
        ###############################
        #Tensorflow
        self.sess = tf.Session()
        with tf.variable_scope('Model') as model_variables:
            self.GAN()
            self.model_var = [var for var in tf.global_variables() if var.name.startswith(model_variables.name)]
            
    
    
    # ===================================================================================================#
    # Tensorflow Functions
    # ===================================================================================================#
    
    def Initialize(self):
        init_model_var = tf.variables_initializer(self.model_var)
        self.sess.run(init_model_var)
        
    # ===================================================================================================#
    # Dataset Functions
    # ===================================================================================================#
    
    def load_datasets(self):
        if self.dataset_name == 'cifar10':
            self.pic = True
            self.data = tf.keras.datasets.cifar10.load_data()
            self.X_data = np.vstack((self.data[0][0],self.data[1][0]))
            self.latent_data = np.vstack((self.data[0][1],self.data[1][1]))
        elif self.dataset_name == 'mnist':
            self.pic = True
            self.data = tf.keras.datasets.mnist.load_data()
            self.X_data = np.expand_dims(np.vstack((self.data[0][0],self.data[1][0])),-1)
            self.latent_data = np.vstack((self.data[0][1].reshape([-1,1]),self.data[1][1].reshape([-1,1])))

    def convert_to_tuples(self):
        self.data = [(self.X_data[i,:,:,:],self.latent_data[i,:],self.Noise_data[i,:]) for i in range(self.X_data.shape[0])]
            
    
    # ===================================================================================================#
    # GAN Functions
    # ===================================================================================================#
    
    ###############################
    #Algorithm
    def GAN(self):
        if self.pic:
            with tf.name_scope('inputs'):
                '''
                self.Noise = tf.random_uniform([self.batch_len,self.Noise_size],
                    minval = 0, maxval=1, name= 'Z')
                '''
                self.Noise= tf.placeholder(tf.float32,[self.batch_len, self.Noise_size], name='Z')
                self.Y = tf.placeholder(tf.float32, [self.batch_len, self.X_height,
                    self.X_width,self.X_debth], name='labels')
                self.latent = tf.placeholder(tf.int32,[self.batch_len, self.latent_input_size], name='labels')
                self.latent_onehot = tf.squeeze(tf.one_hot(self.latent,self.num,dtype=tf.float32))
                self.Z = tf.concat([self.Noise,self.latent_onehot],axis = 1)
                
            with tf.variable_scope('Generator') as gen_variables:
                self.Generator()
                self.G_var = [var for var in tf.trainable_variables() if var.name.startswith(gen_variables.name)]
            with tf.variable_scope('Discriminator') as disc_variables:
                self.Discriminator(True)
                disc_variables.reuse_variables()
                self.Discriminator(False)
                self.D_var = [var for var in tf.trainable_variables() if var.name.startswith(disc_variables.name)]
            with tf.name_scope('Compute_Loss'):
                self.Loss()
            with tf.variable_scope('Optimize'):
                self.Optimize()
                
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('logs/', self.sess.graph)

    ###############################
    #GAN Subcomponents
    
    def Generator(self):
        self.output_gen = self.DCNN(self.Z)
    
    def Discriminator(self,real):
        if real:
            with tf.name_scope('Real_Discriminator'):
                self.output_disc_real,self.output_class_real = self.CNN(self.Y)
                tf.summary.image('Real_Image', self.Y,3)
        else:
            with tf.name_scope('Gen_Discriminator'):
                self.output_disc_gen,self.output_class_gen = self.CNN(self.output_gen)
                tf.summary.image('GAN_Generated_Image', self.output_gen,3)   
    

    # ===================================================================================================#
    # Loss Functions
    # ===================================================================================================#
    def loss_fnc_softmax(self,logits,labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = logits))

    def loss_fnc_sigmoid(self,logits,labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels,logits = logits))  
        
    def Loss(self):
        with tf.name_scope('Class_Loss'):
            with tf.name_scope('Generated_Class_Loss'):
                self.G_Class_loss = self.loss_fnc_softmax(self.output_class_gen,self.latent_onehot)
            with tf.name_scope('Real_Class_Loss'):
                self.R_Class_loss = self.loss_fnc_softmax(self.output_class_real,self.latent_onehot)
                
        with tf.name_scope('Generator_Loss'):
            self.G_loss = self.loss_fnc_sigmoid(self.output_disc_gen,tf.ones_like(self.output_disc_gen))+self.G_Class_loss
            
        with tf.name_scope('Discriminator_Loss'):
            with tf.name_scope('Real'):
                self.D_loss_real = self.loss_fnc_sigmoid(self.output_disc_real,tf.ones_like(self.output_disc_real))+self.R_Class_loss
            with tf.name_scope('Fake'):
                self.D_loss_fake = self.loss_fnc_sigmoid(self.output_disc_gen,tf.zeros_like(self.output_disc_gen))+self.G_Class_loss
            self.D_loss = self.D_loss_fake+self.D_loss_real
            
        tf.summary.scalar('Generator_Loss', self.G_loss)
        tf.summary.scalar('Discriminator_Real_Loss', self.D_loss_real)
        tf.summary.scalar('Discriminator_Fake_Loss', self.D_loss_fake)

    # ===================================================================================================#
    # Optimization Functions
    # ===================================================================================================#
    def Optimize(self):
        self.G_train_step = tf.train.AdamOptimizer(self.lr).minimize(self.G_loss,var_list= self.G_var)
        self.D_train_step_real = tf.train.AdamOptimizer(self.lr).minimize(self.D_loss_real,var_list= self.D_var)
        self.D_train_step_fake = tf.train.AdamOptimizer(self.lr).minimize(self.D_loss_fake,var_list= self.D_var)
        '''
        self.G_gradients = tf.train.AdamOptimizer(self.lr).compute_gradients(self.G_loss,var_list= self.G_var)
        self.Dr_gradients = tf.train.AdamOptimizer(self.lr).compute_gradients(self.D_loss_real,var_list= self.D_var)
        self.Df_gradients = tf.train.AdamOptimizer(self.lr).compute_gradients(self.D_loss_fake,var_list= self.D_var)
        '''
    
    # ===================================================================================================#
    # Learning Functions
    # ===================================================================================================#
    
    def Learn(self):
        for batch in (range(self.batch_size)):
            time.sleep(.02)
            data = random.sample(self.data,self.batch_len)
            data = np.array([i for i in map(list,data)])
            Y = np.stack(data[:,0])
            latent = np.stack(data[:,1])
            Noise = np.stack(data[:,2])
            feed_dict = {self.Noise: Noise,self.Y:Y,self.latent:latent}
            self.sess.run(self.G_train_step,feed_dict=feed_dict)
            self.sess.run(self.D_train_step_real,feed_dict=feed_dict)
            self.sess.run(self.D_train_step_fake,feed_dict=feed_dict)
            self.results = self.sess.run(self.merged,feed_dict = feed_dict)  
            self.writer.add_summary(self.results,self.i)
            self.i += 1
        #self.batch_bar.finish()
    
    # ===================================================================================================#
    # CNN Functions
    # ===================================================================================================#
    
    ###############################
    #Algorithms
    
    def DCNN(self,x):
        with tf.name_scope('DCNN'):
            o1 = self.add_tensordot_layer(x,4,4,102,1,tf.nn.leaky_relu,True)
            o2 = self.add_deconv_layer(o1,4,4,102,51,2,1,tf.nn.leaky_relu,True)
            o3 = self.add_deconv_layer(o2,8,8,51,25,2,2,tf.nn.leaky_relu,True)
            o4 = self.add_deconv_layer(o3,16,16,25,12,1,3,tf.nn.leaky_relu,True)
            output_gen = self.add_deconv_layer(o4,16,16,12,self.X_debth,2,4,tf.nn.tanh)
        return output_gen
            
    def CNN(self,x):
        with tf.name_scope('CNN'):
            o1 = self.add_conv_layer(x,self.X_debth,12,4,0,2,1,tf.nn.leaky_relu)
            o2 = self.add_conv_layer(o1,12,25,4,0,2,2,tf.nn.leaky_relu,True)
            o3 = self.add_conv_layer(o2,25,51,4,0,2,3,tf.nn.leaky_relu,True)
            o4 = self.add_conv_layer(o3,51,102,4,0,1,4,tf.nn.leaky_relu,True)
            o4 = tf.reshape(o4,[self.batch_len,-1])
            o5 = self.add_fc_layer(o4,o4.get_shape()[-1].value,100,1,tf.nn.relu,True)
            output_disc = self.add_fc_layer(o5,o5.get_shape()[-1].value,1,2,tf.nn.sigmoid,False)
            output_class = self.add_fc_layer(o5,o5.get_shape()[-1].value,self.num,3,tf.nn.softmax,False)
        return output_disc,output_class
    
    ###############################
    #Layers
    
    def add_deconv_layer(self,inputs,height,width,feature_old,feature_new,strides,
            n_layer,activation_function=None,batch_normalization = False,unpool = False):
        
        layer_name = 'DeConv_Layer_%s' % n_layer 
        with tf.variable_scope(layer_name):
            with tf.name_scope('Weights'):
                W = self.weights([height,
                    width,feature_new, feature_old], name='W')
            tf.summary.histogram('Weights',W)
            with tf.name_scope('Biases'):
                b = self.bias([feature_new], name='b')
            tf.summary.histogram('Biases',b)
            with tf.name_scope('DeConvolution'):
                output_shape = tf.stack([self.batch_len,width*strides,
                    height*strides,feature_new])
                deconv = tf.nn.conv2d_transpose(inputs,W,
                    strides = [1,strides,strides,1],output_shape=output_shape,padding = 'SAME')
                tf.summary.histogram( 'Output', deconv)
            with tf.name_scope('DeConv_plus_b'):
                if batch_normalization:
                    with tf.name_scope('Batch_Normalization'):
                        output =  tf.contrib.layers.batch_norm(deconv+b)
                        tf.summary.histogram( 'Output', output)
                else:
                    output = deconv+b
                '''
                if unpool:
                    with tf.name_scope('Pooling'):
                        output = tf.nn.max_pool(output,ksize = [1,kernal, kernal,1],
                            strides = [1,kernal,kernal,1],padding = 'SAME') 
                        tf.summary.histogram('Output', output)
                '''
                if activation_function is not None:
                    with tf.name_scope('Activation_Function'):
                        output =  activation_function(output)
                        tf.summary.histogram( 'Output', output) 
        return output 
    
    def add_conv_layer(self,inputs,feature_old,feature_new,patch,kernal,strides,
            n_layer,activation_function=None,batch_normalization = False,pool = False):
        
        layer_name = 'Conv_Layer_%s' % n_layer        
        with tf.variable_scope(layer_name):
            with tf.name_scope('Weights'):
                W = self.weights([patch,patch,feature_old, feature_new], name='W')
            tf.summary.histogram('Weights',W)
            with tf.name_scope('Biases'):
                b = self.bias([feature_new], name='b')
            tf.summary.histogram('Biases',b)
            with tf.name_scope('Convolution'):
                conv = tf.nn.conv2d(inputs,W,strides = [1,strides,strides,1],padding = 'SAME')
                tf.summary.histogram( 'Output', conv)
            with tf.name_scope('Conv_plus_b'):
                if batch_normalization:
                    with tf.name_scope('Batch_Normalization'):
                        output =  tf.contrib.layers.batch_norm(conv+b)
                        tf.summary.histogram( 'Output', output)
                else:
                    output = conv+b
                if pool:
                    with tf.name_scope('Pooling'):
                        output = tf.nn.max_pool(output,ksize = [1,kernal, kernal,1],
                            strides = [1,kernal,kernal,1],padding = 'SAME') 
                        tf.summary.histogram('Output', output)
                if activation_function is not None:
                    with tf.name_scope('Activation_Function'):
                        output =  activation_function(output)
                        tf.summary.histogram( 'Output', output)                     
        return output 
    
    # ===================================================================================================#
    # Dense Functions
    # ===================================================================================================#

    def add_tensordot_layer(self,inputs,width,height,feature_new,
            n_layer,activation_function=None, keep_prob_ = False):
        
        layer_name = 'Tensordot_Layer_%s' % n_layer
        with tf.name_scope(layer_name):
            with tf.name_scope('Weights'):
                W = self.weights([inputs.get_shape()[-1],width,height, feature_new], name='W')
            tf.summary.histogram('Weights',W)
            with tf.name_scope('Biases'):
                b = self.bias([feature_new], name='b')
            tf.summary.histogram('Biases',b)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.tensordot(inputs,W,axes = 1)+b
                if activation_function is not None:
                    Wx_plus_b = activation_function(Wx_plus_b)
                tf.summary.histogram( 'Output', Wx_plus_b)
            with tf.name_scope('Dropout'):
                if keep_prob_ == True:
                    outputs = tf.nn.dropout(Wx_plus_b,self.keep_prob)
                    tf.summary.histogram( 'Output', outputs)
                else:
                    outputs = Wx_plus_b
        return outputs    
    
    def add_fc_layer(self,inputs, in_size, out_size, n_layer,activation_function=None, keep_prob_ = False):

        layer_name = 'FC_Layer_%s' % n_layer
        with tf.variable_scope(layer_name):
            with tf.name_scope('Weights'):
                Weights = self.weights([in_size, out_size], name='W')
            tf.summary.histogram( 'Weights',Weights)
            with tf.name_scope('Biases'):
                biases = self.bias([out_size],name='b')
            tf.summary.histogram('Biases',biases)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
                if activation_function is not None:
                    Wx_plus_b = activation_function(Wx_plus_b)
                tf.summary.histogram( 'Output', Wx_plus_b)
            with tf.name_scope('Dropout'):
                if keep_prob_ == True:
                    outputs = tf.nn.dropout(Wx_plus_b,self.keep_prob)
                    tf.summary.histogram( 'Output', outputs)
                else:
                    outputs = Wx_plus_b
        return outputs

    # ===================================================================================================#
    # Parameter Variables
    # ===================================================================================================#

    def weights(self, shape, name):
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def bias(self, shape, name):
        initializer = tf.constant_initializer(.1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)
    
#############################################################################################################################################################################################################################
# %% Run Code
    
if __name__ == '__main__':
    
    tf.reset_default_graph()
    epoch_bar = progressbar.ProgressBar()

    md = Model()
    md.Initialize()
    
    for epoch in epoch_bar(range(md.epoch_size)):
        time.sleep(.02)
        md.Learn()
    
    
    