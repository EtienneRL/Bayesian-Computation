# -*- coding: utf-8 -*-
"""
Created on Fri May 18 21:04:41 2018

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
    
def BacktrackingLSGD(f,theta0,L0,em,g,a,b):
    tn=np.array(theta0)
    e=1
    while e>=em:
        q=scipy.optimize.approx_fprime(tn,f,g)
        L=L0
        e=(q*q).sum()
        t=1/2*L*a*e
        while t<L*a*e:
            tn1=tn+L*q
            t=abs(f(tn1)-f(tn))
            L=b*L
        tn=tn1
    return tn

def LaplaceApproximation(f,theta0,L0,em,g,a,b,epsilon):
    thetae=BacktrackingLSGD(f,theta0,L0,em,g,a,b)
    beta=-hessian(thetae,logf(f),epsilon)
    return [thetae,beta]

def RSSamplingGaussian(f,theta0,L0,em,g,a,b,epsilon,K,n): #f function #(theta0,L0,em,g,a,b) arguments for BTLS gd #epsilon step for hessian #K factor for hypograph func #n number of samples
    m=LaplaceApproximation(f,theta0,L0,em,g,a,b,epsilon)
    print(m)
    v=scipy.stats.multivariate_normal(m[0],m[1])
    A=[]
    for i in range(n):
        thetatest=np.random.multivariate_normal(m[0],m[1])
        u=np.random.uniform(0,1)
        j=v.pdf(thetatest)
        u1=f(thetatest)/(K*j)
        print(u1)
        if u<=u1:
            A+=[thetatest]
    print(len(A))
    return A

print(RSSamplingGaussian(somegaussiandensity,[2,1],0.3,0.000001,0.0001,1/2,1/2,0.0001,1.5,400))

#careful: gradient descent must be efficient or failure #try and search for a good initial point for gradient descent
#We could do better we could compute a K