from PIL import Image as img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import random
import math


def g(z):
    """ SIGMOID FUNCTION """

    return 1/(1+np.exp(-z))





def J(g,y):
    """ COST FUNCTION """

    g[g<=0]=0.000000001
    g[g>=1]=0.999999999
    return ((y @ np.log(g).transpose()+(1-y) @ np.log(1-g).transpose())/(-len(y)))[0,0]




def gd(theta,x,y,alpha=0.00025,itr=2000,graph=False,cost=False):
    """ GRADIENT DESCENT FUNCTION """

    if graph:
        cost = []
        for i in range(itr):
            theta = theta - (x @ (g(theta.transpose() @ x)-y).transpose() * alpha/y.size)
            c = J(g(theta.transpose()@x),y)
            cost += [c]
            if cost and i%500==0:
                print(f"After {i} iterations cost : {c}")
        # plt.open
        plt.close()
        plt.plot(np.arange(itr),cost)
        plt.show()
    else:
        for i in range(itr):
            theta = theta - x @ (g(theta.transpose() @ x)-y).transpose() * alpha/y.size
    return theta



def predict(x,y,theta):
    predict_y = g(theta.T @ x)
    predict_y[predict_y>=0.5]=1
    predict_y[predict_y<0.5]=0
    # print(predict_y==y)
    return len(predict_y[predict_y==y])*100/m
