# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:21:37 2018

@author: étienne
"""
import xlrd
import numpy as np
import math
import scipy
from numpy import array
from scipy import stats
from numpy import random
from numpy import linalg


def gettable(a,k):  #A='excel.xlxs' k is index of sheet we want to get (0 for firsto one)
    x=xlrd.open_workbook(a)
    y=x.sheet_by_index(k)
    n=y.nrows
    A=[]
    for i in range(n):
        A+=[y.row_values(i)]
    return A

def indextoteam(T,o,p): #T is table from gettable #o column no of team 1 names #p column no of team 1 names
    n=len(T)
    B=[[T[0][o],0]]
    c=0
    for i in range(n):
        j=0
        a=0
        while j<len(B) and a==0:
            if B[j][0]==T[i][o]:
                a=1
            j+=1
        if a==0:
            c+=1
            B+=[[T[i][o],c]]
    for i in range(n):
        j=0
        a=0
        while j<len(B) and a==0:
            if B[j][0]==T[i][p]:
                a=1
            j+=1
        if a==0:
            c+=1
            B+=[[T[i][p],c]]
    return B
#len(B)=number of teams
    
def formatdatacol(T,a,b,indcol): #indcol list of index of columns to add #a column no of team 1 names #b column no of team 2 names 
    n=len(T)
    B=indextoteam(T,a,b)
    A=[]
    for i in range(n):
        K=[0,0]
        a1=0
        a2=0
        j=0
        while (a1==0 or a2==0):
            if B[j][0]==T[i][a]:
                a1=1
                K[0]=B[j][1]
            if B[j][0]==T[i][b]:
                a2=1
                K[1]=B[j][1]
            j+=1
        for e in range(len(indcol)):
            K+=[T[i][indcol[e]]]
        A+=[K]
    return A

def formatdatarow(T,excluderow): #excluderowe: exclude row j for i=excluderow[0] T[i][j] is in excluderow[1]
    n=len(T)
    m=len(excluderow)
    T1=[]
    for i in range(n):
        a=0
        for j in range(m):
            e=excluderow[j]
            for k in range(len(e[1])):
                if T[i][e[0]]==e[1][k]:
                    a=1
        if a==0:
            T1+=[T[i]]
    return T1

def Gaussian_parametric_model(T,m):
    def f_d_theta(theta):
        x=1
        mu0=theta[m]
        sigma=theta[m+1]
        for k in range(len(T)):
            i=T[k][0]
            j=T[k][1]
            fi=(1/(2*math.pi*(sigma**2)))**(1/2)*math.exp(-1/2*(((T[k][2]-(mu0+theta[i]+theta[j]))/sigma)**2))*100
            x=fi*x
        return x
    return f_d_theta

def gaussian_f_prior_gm(mu,mu0,sigma0,v,v0,vsigma0,m):
    mean=[0]*(m+2)
    for i in range(m):
        mean[i]=mu
    mean[m]=mu0
    mean[m+1]=sigma0
    var=v*np.identity(m+2)
    var[m][m]=v0
    var[m+1][m+1]=vsigma0
    def k(theta):
        b=scipy.stats.multivariate_normal(mean,var)
        a=b.pdf(theta)
        return a
    return k

def Gaussian_posterior(D,m,prior):#m number of teams for logit_parametric_model AND NOT number of parameters
    f1=Gaussian_parametric_model(D,m)
    f2=prior
    def f(theta):
        x=f1(theta)*f2(theta)
        return x
    return f

I=gettable('AllLeagues.xlsx',2)
U=formatdatarow(I,[[1,[2014,2015,2016,2018]],[3,['promotion','regional']]])
J=indextoteam(U,4,7)
m=len(J)
K=formatdatacol(U,4,7,[8])
sigma0=10#7.32
vsigma0=3
mu=0
mu0=37
v=2
v0=5

n1=len(K)
K1=[]
K2=[]
for i in range(n1):
    if i%11==0:
        K2+=[K[i]]
    else:
        K1+=[K[i]]

ft=gaussian_f_prior_gm(mu,mu0,sigma0,v,v0,vsigma0,m)
ftd=Gaussian_posterior(K1,m,ft)

def ISSampling(f,hproposal,S,n): #f posterior density #hproposal density from which we first sample theta (simple for a gaussian) #S function we compute the mean #n number of samples
    A=0
    B=0
    v=scipy.stats.multivariate_normal(hproposal[0],hproposal[1])
    for i in range(n):
        thetai=np.random.multivariate_normal(hproposal[0],hproposal[1])
        pi=f(thetai)/v.pdf(thetai)
        A+=pi*S(thetai)
        B+=pi
        print(i)
    E=A/B
    return E

B=[0]*(m+2)
B[m]=35
B[m+1]=3
def S(x):
    return x
hp=[B,np.identity(m+2)]
a=ISSampling(ftd,hp,S,15000)

def gaussian_prediction(theta_estimate,T):
    A=[]
    m=len(theta_estimate)
    for k in range(len(T)):
        i=T[k][0]
        j=T[k][1]
        tij=theta_estimate[m-2]+theta_estimate[i]+theta_estimate[j]
        A+=[tij]
    A=np.array(A)
    return A

def Gaussian_MSE(theta_estimate,T):
    MSE=0
    m=len(theta_estimate)
    n=len(T)
    for k in range(len(T)):
        i=T[k][0]
        j=T[k][1]
        MSE+=(T[k][2]-(theta_estimate[m-2]+theta_estimate[i]+theta_estimate[j]))**2
    MSE=MSE/n
    return MSE

def calculate_var_data(T):
    n=len(T)
    m=0
    for i in range(n):
        m+=T[i][2]
    m=m/n
    v=0
    for i in range(n):
        v+=(T[i][2]-m)**2
    v=v/n
    return v

print(Gaussian_MSE(a,K2))
print(calculate_var_data(K2))
print(a)

#[ 0.27146269 -1.33940752  1.4676984   0.24886374 -0.30738779  1.06795843
# -1.82426051  0.84368783  1.53251995  0.89689695 -0.44767283 -0.68846325
#  0.73365868 -0.10116423 -0.41294957 -0.606911   36.11247382  6.73007374]