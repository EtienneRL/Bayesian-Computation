# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:45:51 2018

@author: étienne
"""
import xlrd
import numpy as np
import math
import scipy as sc
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
        b=sc.stats.multivariate_normal(mu,cov)
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

I=gettable('AllLeagues.xlsx',2)
U=formatdatarow(I,[[1,[2014,2016,2017,2018]],[2,['Spring']],[3,['promotion','regional']]])
J=indextoteam(U,4,7)
m=len(J)
K=formatdatacol(U,4,7,[5])
mu_prior=[0]*(m+1)
p=6
cov_prior=p*np.identity(m+1)
ft=gaussian_f_prior(mu_prior,cov_prior)
ftd=logit_posterior(K,m,ft)
#ftd=sc.stats.multivariate_normal([0]*(m+1), 4*np.identity(m+1)).pdf

def logf(f):
    def logf1(x):
        a=math.log(f(x))
        return a
    return logf1
    
def hessian (x,f,e):
    n=len(x)
    x=np.array(x)
    a=sc.optimize.approx_fprime(x,f,e)
    m=np.zeros(n)
    A=[]
    for i in range(n):
        k=m
        k[i]=e
        a1=sc.optimize.approx_fprime(x+k,f,e)
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
        q=sc.optimize.approx_fprime(tn,f,g)
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
    return tn

def LaplaceApproximation(f,theta0,L0,em,g,a,b,epsilon,nmax):
    thetae=BacktrackingLSGD(logf(f),theta0,L0,em,g,a,b,nmax)
    beta=np.linalg.inv(-hessian(thetae,logf(f),epsilon))
    return[thetae,beta]

theta0=[0]*(m+1)
L0=0.01
em=0.00001
g=0.0000000000001
a=1/2
b=1/10
epsilon=0.00001
nmax=40
lftd=logf(ftd)
A=LaplaceApproximation(ftd,theta0,L0,em,g,a,b,epsilon,nmax)
mg=sc.stats.multivariate_normal(A[0],A[1])

a0=[-0.0067224 , -0.34317549, -0.50061066,  0.89145913, -1.02628461,  1.21185284,       0.36661785,    2.57155963,  -0.15348833,  -0.52191584, -1.18025034, -1.28506095,  0.06434853]
x=np.linspace(-4,4,200)
for j in range(len(a0)):
    a=a0
    y1=[]  
    y2=[]
    k=a0[j]
    for i in range(len(x)):
        a[j]=x[i]
        u=ftd(a)/(6.2e10)#/(2.34**(len(K)/2.85))
        y1+=[u]
        y2+=[mg.pdf(a)]
        a[j]=k
        print(a0)
    a0[j]=k
    y1=np.array(y1)
    y2=np.array(y2)
    if j==m:
        plt.plot(x,y1)
        plt.plot(x,y2)
plt.show()