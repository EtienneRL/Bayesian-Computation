# -*- coding: utf-8 -*-
"""
Created on Thu May 17 21:57:01 2018

@author: étienne
"""
import numpy as np
import scipy
from scipy import optimize


def ex(x):
    a=-1/2*x[0]**2-1/4*(x[1]-1)**2
    return a

def BacktrackingLSGD(f,theta0,L0,em,g,a,b,nmax):
    tn=np.array(theta0)
    e=1
    while e>=em:
        q=scipy.optimize.approx_fprime(tn,f,g)
        L=L0
        e=(q*q).sum()
        tn1=tn+L*q
        n=0
        while f(tn1)<=f(tn)+abs(L*a*e):
            L=b*L
            tn1=tn+L*q
            n+=1
            if n==nmax:
                return tn1
        tn=tn1
    return tn

print(BacktrackingLSGD(ex,[1,2],0.3,0.001,0.1,1/2,1/10))
