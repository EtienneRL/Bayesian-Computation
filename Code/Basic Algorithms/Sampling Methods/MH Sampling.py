# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:30:40 2018

@author: étienne
"""
import math
import scipy
import numpy as np
from scipy import stats
from numpy import random
from numpy import linalg

def somegaussiandensity(x):
    sigma=np.identity(len(x))
    mu=[]
    for i in range(len(x)):
        mu+=[i**2]
    f=scipy.stats.multivariate_normal(mu,sigma)
    a=f.pdf(x)
    return a

def MHSampling(f,theta0,L,n):
    tn=np.array(theta0)
    N=len(theta0)
    L=np.array(L)
    mu=np.array([0]*N)
    cov=np.identity(N)
    A=[]
    for i in range(n):
        a=L*np.random.multivariate_normal(mu,cov) #Here I chose to use a Gaussian
        tp=tn+a
        k=f(tp)/f(tn)
        if k>=1:
            tn=tp
            A+=[tn]
        else: 
            b=np.random.uniform(0,1)
            if b<=k:
                tn=tp
            A+=[tn]
    return A

x=[1,15,-3,-4,5]
L=MHSampling(somegaussiandensity,x,0.08,100000)  
A=np.zeros(len(x))
for i in range(int(len(L)/2)):
    A+=np.array(L[2*i])
A=A/int(len(L)/2)
print(A)
