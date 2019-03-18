# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:12:20 2018

@author: étienne
"""
import math
import scipy
import numpy as np

def somegaussiandensity(x):
    sigma=math.sqrt(4)
    mu=[3.14,-0.542]
    a=1/(((2*math.pi)**(1/2))*sigma)*math.exp(-1/2*((x[0]-mu[0])/sigma)**2-1/2*((x[1]-mu[1])/sigma)**2)
    return a

def logf(f):
    def logf1(x):
        a=math.log(f(x))
        return a
    return logf1
    
def hessian (x,f,e):
    n=len(x)
    x=np.array(x)
    a=scipy.optimize.approx_fprime(x,f,e)
    m=np.zeros(n)
    A=[]
    for i in range(n):
        k=m
        k[i]=e
        a1=scipy.optimize.approx_fprime(x+k,f,e)
        k[i]=0
        a2=(a1-a)/e
        A+=[a2]
    A=np.array(A)
    return A

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

def LaplaceApproximation(f,theta0,L0,em,g,a,b,epsilon,nmax):
    thetae=BacktrackingLSGD(logf(f),theta0,L0,em,g,a,b,nmax)
    beta=np.linalg.inv(-hessian(thetae,logf(f),epsilon))
    return[thetae,beta]
    
print(LaplaceApproximation(somegaussiandensity,[4,-1/2],0.3,0.000000000000000000001,0.001,1/2,1/2,0.001,200000))
