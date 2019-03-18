# -*- coding: utf-8 -*-
"""
Created on Wed May 23 23:02:22 2018

@author: étienne
"""
import numpy as np

def logit2_posterior(D,m1,m2,prior):#m number of teams for logit_parametric_model
    f1=logit2_parametric_model(D,m1,m2)
    f2=prior
    def f(theta):
        x=f1(theta)*f2(theta)
        return x
    return f

I=gettable('AllLeagues.xlsx',1)
B=indextoteam(I,4,7)
m=len(B)
D=formatdatacol(I,4,7,[5])
p=gaussian_f_prior([0]*(m+1),np.identity(m+1))

logit_posterior([0]*(m+1),D,p)
