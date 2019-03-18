# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:08:51 2018

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
import matplotlib.pyplot as plt


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

def logit1_parametric_model(T,m): #T[k][0]=i T[k][1]=j T[k][2]=yij #m number of teams
    def f_d_theta(theta):
        x=1
        for p in range(len(T)):
            i=T[p][0]
            j=T[p][1]
            y=T[p][2]
            pij=1/(1+math.exp(-theta[i]+theta[j]-theta[m]))
            if y==1:
                f_d_thetai=pij
            else:
                f_d_thetai=1-pij
            x=f_d_thetai*x*2.34 #here the factor aims to have a posterior in a good range of values (not 0)
        return x
    return f_d_theta

def gaussian_f_prior(mu,cov):
    def k(theta):
        b=scipy.stats.multivariate_normal(mu,cov)
        a=b.pdf(theta)
        return a
    return k

def logit_posterior(D,m,prior):#m number of teams for logit_parametric_model AND NOT number of parameters
    f1=logit1_parametric_model(D,m)
    f2=prior
    def f(theta):
        x=f1(theta)*f2(theta)
        return x
    return f

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
    A=[]
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
        A+=[tn1]
        print(A)
    return tn

def LaplaceApproximation(f,theta0,L0,em,g,a,b,epsilon,nmax):
    thetae=BacktrackingLSGD(f,theta0,L0,em,g,a,b,nmax)
    beta=np.linalg.inv(-hessian(thetae,logf(f),epsilon))
    return[thetae,beta]

I=gettable('AllLeagues.xlsx',2)
M=[[[1,[2014,2016,2017,2018]],[2,['Spring']],[3,['promotion','regional']]],[[1,[2014,2016,2017,2018]],[2,['Summer']],[3,['promotion','regional']]],[[1,[2014,2015,2017,2018]],[2,['Spring']],[3,['promotion','regional']]],[[1,[2014,2015,2017,2018]],[2,['Summer']],[3,['promotion','regional']]],[[1,[2014,2016,2015,2018]],[2,['Spring']],[3,['promotion','regional']]],[[1,[2014,2016,2015,2018]],[2,['Summer']],[3,['promotion','regional']]]]
X1=range(len(M))
X2=[]
for i in range(len(M)):
    U=formatdatarow(I,M[i])
    J=indextoteam(U,4,7)
    m=len(J)
    K=formatdatacol(U,4,7,[5])
    
    n1=len(K)
    K1=[]
    K2=[]
    for i in range(n1):
        if i%10==0:
            K2+=[K[i]]
        else:
            K1+=[K[i]]
    mu_prior=[0]*(m+1)
    p=6
    cov_prior=p*np.identity(m+1)
    ft=gaussian_f_prior(mu_prior,cov_prior)
    ftd=logit_posterior(K1,m,ft)
    #ok we got the posterior now
    #now let's sample using Laplace Approximation
    
    theta0=[0]*(m+1)
    L0=0.01
    em=0.00001
    g=0.0000000000001
    a=1/2
    b=1/10
    epsilon=0.0001
    nmax=40
    lftd=logf(ftd)
    A=LaplaceApproximation(lftd,theta0,L0,em,g,a,b,epsilon,nmax)
    a=A[0]
    
    
    u=[]
    for i in range(len(J)):
        u+=[[J[i],a[i]]]
    print(u)
    
    
    def logit_prediction(theta_estimate,T):
        A=[]
        m=len(theta_estimate)
        for k in range(len(T)):
            i=T[k][0]
            j=T[k][1]
            pij=theta_estimate[i]-theta_estimate[j]+theta_estimate[m-1]
            c=0
            if pij>=0:
                c=1
            if c==T[k][2]:
                A+=[1]
            else:
                A+=[0]
        A=np.array(A)
        c=np.sum(A)/len(A)*100
        return c
    
    b=logit_prediction(a,K2)
    X2+=[b]
plt.plot(X1,X2)
plt.show
    #[-0.0067224 ,           -0.34317549,          -0.50061066,             0.89145913,     -1.02628461,        1.21185284,          0.36661785,        2.57155963,         -0.15348833,           -0.52191584,           -1.18025034,           -1.28506095,      ##0.06434853]
    #[0.18769645589412576, -0.25426223241495083, -0.40064658218089505, 1.1616403630706256, -0.9287594781265749, 1.5245897940976816, 0.5857796072577182, 3.089441104415934, -0.0011963880555859781, -0.39268043159406496, -1.4311497019907375, -2.5742139604702023]