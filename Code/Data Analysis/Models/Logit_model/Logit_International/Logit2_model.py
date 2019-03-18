# -*- coding: utf-8 -*-
"""
Created on Wed May 23 22:47:51 2018

@author: étienne
"""
import math

def logit2_parametric_model(T,m1,m2): #T[k][0]=i T[k][1]=j T[k][2]=yij #m1 number of teams #T[k][3/4] region of i/j
    def f_d_theta(theta): #len(theta)=m1+m2+1 (force of teams force of regions blue side bias)
        x=1
        for p in range(len(T)):
            i=T[p][0]
            j=T[p][1]
            k=T[p][2]
            l=T[p][3]
            y=T[p][4]
            pij=1/(1+math.exp(-theta[i]+theta[j]-theta[m1+k]+theta[m1+l]-theta[m1+m2]))
            if y==1:
                f_d_thetai=pij
            else:
                f_d_thetai=1-pij
            x=f_d_thetai*x*2.34 #here the factor aims to have a posterior in a good range of values (not 0)
        return x
    return f_d_theta

I=gettable('AllLeagues.xlsx',1)
n=len(I)
B=indextoteam(I,4,7)
C=formatdatacol(I,4,7,[5])
E=formatdatarow(C,[[0,range(6,n)]])
f=logit_parametric_model(C)
n=len(C)
f0=f([1]*(n+1))
print(f0)

