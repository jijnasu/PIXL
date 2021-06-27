from PIL import Image as img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import random
import math

def resize_rename_img(datafolder="raw_data",target="data",height=250,width=250):
    """
    arguments:
        datafolder  : Folder cotaining raw data(images)
        target      : Target folder to save prcessed data
        height      : Height of image to be saved
        width       : Width of image to be saved
    result:
        -> Cleaning target folder
        -> Processing images and saving to target
    return:
        None
    """
    
    # Cleaning target folder
    gun_names = glob.glob(target+"\\*.jpeg")
    gun_names += glob.glob(target+"\\*.jpeg")
    for gun in gun_names:
        os.unlink(gun)
    
    # Processing and saving to target
    gun_names = glob.glob(datafolder+"\\*.jpeg")
    gun_names += glob.glob(datafolder+"\\*.jpeg")
    i=0
    for gun in gun_names[:]:
        gun_img = img.open(gun)
        shp = np.array(gun_img).shape
        if gun_img.height>=height and gun_img.width>=width and shp[-1]==3:
            print("hello")
            gun_img.resize((width,height)).save("data\\gun"+str(i)+".jpg")
            i+=1
    print("Total no of images processed and saved : ",i)




def label_data(datafolder="data"):
    """
    arguments:
        datafolder  : Folder cotaining processed data(images)
    result:
        -> Takes lable for each image as:
                -   '1': if image is having gun
                -   '0': if image is not having gun
    return:
        all_data_y  : numpy array of lables
    """
    gun_names = glob.glob(f"{datafolder}\*.jpg")
    all_data_y=[]
    i=0
    for name in gun_names[:]:
        i+=1
        gun = img.open(name)
        plt.imshow(gun)
        plt.show()
        # gun = np.array(gun).reshape(1,n)
        ty = ''
        while ty not in [0,1]:
            ty = (input(f"{i}. Is this a Gun(0 or 1)? "))   # Takes lable for each image(i.e. 1 or 0)
            if ty in ['0','1']:
                ty = int(ty)
        plt.close()
        all_data_y+=[ty]
    all_data_y = np.array(all_data_y)
    # m = len(all_data_y)
    return all_data_y





def save_all_data_y(all_data_y,gun_lables = 'gundatay.csv'):
    """
    arguments:
        all_data_y  : array of lables
        gun_lables  : target file name
    result:
        saves all_data_y to gun_lables
    return:
        None
    """
    np.savetxt(gun_lables,all_data_y.reshape(1,all_data_y.size),delimiter=',')





def check_rectify(datafolder="data",gun_lables="gundatay.csv"):
    """
    arguments:
        datafolder  : Folder cotaining processed data(images)
        gun_lables  : file containing all data lables (i.e. all_data_y)
    result:
        -> prints all images with its saved labels one by one and tales input as:
                -   '' : if it's correct
                -   '0': if image is not having gun
                -   '1': if image is having gun
        -> saves all lables to the same file
    return:
        None
    """
    
    # Checking and rectify
    gun_names = glob.glob(f"{datafolder}\*.jpg")
    all_data_y = np.genfromtxt(gun_names,delimiter=',')
    for i,name in enumerate(gun_names[:]):
        gun = img.open(name)
        plt.imshow(gun)
        plt.show()
        ip = 120
        while ip not in ['0','1','']:
            ip = (input(f"{i}. Lable : {all_data_y[i]}"))
            if ip in ['0','1']:
                all_data_y[i] = int(ip)

    # Saving to the same file
    save_all_data_y(all_data_y,gun_lables)


resize_rename_img("raw_data","data")