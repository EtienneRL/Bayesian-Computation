# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:22:01 2018

@author: étienne
"""
import math
import scipy
import numpy as np
from scipy import stats
from numpy import random
from numpy import linalg

def somegaussiandensity(x):
    sigma=1
    mu=[0,0]
    a=1/((2*math.pi)*sigma)*math.exp(-1/2*((x[0]-mu[0])/sigma)**2-1/2*((x[1]-mu[1])/sigma)**2)
    return a

def S(x): #fonction dont on veut calculer l'espérance par sampling
    a=x**2
    return a

def ISSampling(f,hproposal,S,n): #f posterior density #hproposal density from which we first sample theta (simple for a gaussian) #S function we compute the mean #n number of samples
    A=0
    B=0
    v=scipy.stats.multivariate_normal(hproposal[0],hproposal[1])
    for i in range(n):
        thetai=np.random.multivariate_normal(hproposal[0],hproposal[1])
        pi=f(thetai)/v.pdf(thetai)
        A+=pi*S(thetai)
        B+=pi
    E=A/B
    return E
#we can use Laplace for hproposal        
        
print(ISSampling(somegaussiandensity,[[0,0],[[1,0],[0,1]]],S,5000))

