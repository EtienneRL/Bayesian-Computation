# -*- coding: utf-8 -*-
"""
Created on Thu May 17 19:12:40 2018

@author: étienne
"""
import numpy as np 
import scipy
from scipy import optimize
import random as rd

def ex(x):
    a=-1/2*x[0]**2-1/4*(x[1]-15)**2
    return a

def StochasticGD(f,theta0,em,g): #f function #theta0 initial point #L setp-size for gd #b error to get #g argument for gradient
    tn=np.array(theta0)
    n=len(theta0)
    e=1
    k=1
    while e>=em:
        i=rd.randint(0,n-1)
        print(i)
        m=np.zeros(n)
        m[i]=g
        tn[i]+=1/k*(f(tn+m)-f(tn))/g #Not very efficient algorithm with that step size: exponential time scale; could do with integer/k with right integer
        m[i]=0
        if k%100==0:
            a=scipy.optimize.approx_fprime(tn,f,np.array([g]*n))
            e=(a*a).sum()
            print(e)
            print(tn)
        k+=1
    return tn
        
StochasticGD(ex,[1.,2.],0.0001,0.01) #1.,2. important or the algorithm only uses integers
