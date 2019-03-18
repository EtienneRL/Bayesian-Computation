# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:28:36 2018

@author: étienne
"""
import math
import scipy
import numpy as np


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

def rdf(x):
    a=1/3*x[0]**3+1/4*x[1]**2
    return a

print(hessian([1,0],rdf,0.001))