# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:03:06 2018

@author: étienne
"""
import numpy as np
import scipy
from scipy import optimize


def ex(x):
    a=-1/2*x[0]**2-1/4*(x[1]-1)**2
    return a

def BasicGD(f,theta0,L,em,g): #f function #theta0 initial point #L setp-size for gd #b error to get #g argument for gradient
    tn=np.array(theta0)
    e=1
    while e>=em:
        a=scipy.optimize.approx_fprime(tn,f,g)
        tn1=tn+L*a
        e=(a*a).sum()
        tn=tn1
    return tn
 
 
        

print(BasicGD(ex,[1,2],0.01,0.0001,[0.01,0.01]))