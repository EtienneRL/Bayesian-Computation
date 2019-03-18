# -*- coding: utf-8 -*-
"""
Created on Sat May 19 18:01:58 2018

@author: étienne
"""
import math
import scipy
import numpy as np
from scipy import stats
from numpy import random
from numpy import linalg
from scipy import optimize

def somegaussiandensity(x):
    sigma=np.identity(len(x))
    mu=[]
    for i in range(len(x)):
        mu+=[i**2]
    f=scipy.stats.multivariate_normal(mu,sigma)
    a=f.pdf(x)
    return a

def logf(f):
    def logf1(x):
        a=math.log(f(x))
        return a
    return logf1

def LangevinMHSampling(f,theta0,L0,n,g): #g step-size for gradient computation
    tn=np.array(theta0)
    N=len(theta0)
    mu=np.array([0]*N)
    cov=np.identity(N)
    f0=logf(f)
    A=[]
    for i in range(n):
        a=L0*np.random.multivariate_normal(mu,cov) #Here I chose to use a Gaussian
        w1=np.array((L0**2)*1/2*scipy.optimize.approx_fprime(tn,f0,g))
        tp=tn+w1+a
        w2=np.array((L0**2)*1/2*scipy.optimize.approx_fprime(tp,f0,g))
        V1=np.array(tp-tn-w1)
        V2=np.array(tn-tp-w2)
        k=f(tp)/f(tn)*math.exp(-1/(2*L0**2)*((V2*V2).sum()-(V1*V1).sum()))
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
L=LangevinMHSampling(somegaussiandensity,x,1,50000,0.0001)  
A=np.zeros(len(x))
for i in range(int(len(L)/2)):
    A+=np.array(L[2*i])
A=A/int(len(L)/2)
print(A)

#We notice that choosing the right step size is much more important than for classic MH