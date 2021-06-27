from PIL import Image as img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import random
import math



def extract(gunimg_file="data",labels_file='gundatay.csv'):
    """
    arguments:
        gunimg     : Folder cotaining raw data(images)
        label_file : y data (data such as 0 & 1 arranged in a vector)
    result:
        -> extracting features such as RGB values to new variables
        -> flattening images to perform vectorization
    return:
        flattened imaged with the no. of training examples & features
    """
    
    gun_names = glob.glob(gunimg_file+"\*.jpg")
    all_data_x=[]
    for name in gun_names[:]:
        gun = img.open(name)
        gun = np.array(gun)
        # print(gun.shape)
        gun = gun.flatten()#1,gun.width*gun.height*3)
        all_data_x += [gun]
    all_data_x = np.array(all_data_x)
    all_data_y = np.genfromtxt(labels_file,delimiter=',')
    n = all_data_x.shape[-1]
    m = all_data_y.size
    all_data_x = all_data_x.reshape(m,n).transpose()
    all_data_y = all_data_y.reshape(1,m)
    all_data_x = np.vstack((all_data_x,np.ones((1,m))))
    n += 1
    return all_data_x,all_data_y,m,n



def shuffle_data(all_data_x,all_data_y,m,n):
    """
    arguments:
        all_data_x : x dataset i.e. images to train 
        all_data_y : y dataset i.e. values of corresponding x datasets
        m          : no. of train examples
        n          : no. of features
    result:
        -> shuffles data in all data x and all data y 
    return:
        shuffled form of data
    """
    
    all_data = np.vstack((all_data_x,all_data_y)).copy()
    all_data = all_data.transpose()
    for i in range(3):
        np.random.shuffle(all_data)
    # print(all_data.shape,all_data.T[-1,:])
    all_data = all_data.transpose()
    all_x = all_data[:-1,:].copy()
    all_y = all_data[-1:,:].copy()
    return all_x,all_y



def divide_train_test(all_data_x,all_data_y,m,n):
    """
    arguments:
        all_data_x : x train dataset i.e. images to train 
        all_data_y : y train dataset i.e. values of corresponding x train datasets
        m          : no. of train examples
        n          : no. of features
    result:
        -> we are dividing the whole data into training set and testing set
    return:
        trained set and test set
    """
    
    no_train = int(m*0.70)
    train_x = all_data_x[:,:no_train]
    test_x = all_data_x[:,no_train:]
    train_y = all_data_y[:,:no_train]
    test_y = all_data_y[:,no_train:]
    return train_x,test_x,train_y,test_y,no_train,(m-no_train)
