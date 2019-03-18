# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 17:32:49 2018

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

def esperance(T):
    a=0
    n=len(T)
    for i in range(n):
        a+=T[i][2]
    a=1/n*a
    return a
    
def mu_sigma(T):
    mu=esperance(T)
    var=0
    n=len(T)
    for i in range(n):
        var+=(T[i][2]-mu)**2
    var=1/n*var
    sigma=var**(1/2)
    return [mu,sigma]

def normalize_data(T,a):
    A=[]
    n=len(T)
    for i in range(n):
        K=[0,0,0]
        K[0]=T[i][0]
        K[1]=T[i][1]
        K[2]=(T[i][2]-a[0])/a[1]
        A+=[K]
    return A

def pdf(x):
    return 1/math.sqrt(2*math.pi) * math.exp(-x**2/2)

def cdf(x):
    return (1 + sc.special.erf(x/math.sqrt(2))) / 2

def skew(x,e=0,w=1,a=0):
    t = (x-e) / w
    return 2 / w * pdf(t) * cdf(a*t)

def Asymetric_Gaussian_parametric_model(T,m):
    def f_d_theta(theta):
        x=1
        mu0=theta[m]
        sigma=theta[m+1]
        a=theta[m+2]
        for k in range(len(T)):
            i=T[k][0]
            j=T[k][1]
            fi=skew(T[k][2],mu0+theta[i]+theta[j],sigma,a)*10
            x=fi*x
        return x
    return f_d_theta   

def gaussian_f_prior_agm(mu,mu0,sigma0,a0,v,v0,vsigma0,va0,m):
    mean=[0]*(m+3)
    for i in range(m):
        mean[i]=mu
    mean[m]=mu0
    mean[m+1]=sigma0
    mean[m+2]=a0
    var=v*np.identity(m+3)
    var[m][m]=v0
    var[m+1][m+1]=vsigma0
    var[m+2][m+2]=va0
    def k(theta):
        b=sc.stats.multivariate_normal(mean,var)
        a=b.pdf(theta)
        return a
    return k

def Asymetric_Gaussian_posterior(D,m,prior):#m number of teams for logit_parametric_model AND NOT number of parameters
    f1=Asymetric_Gaussian_parametric_model(D,m)
    f2=prior
    def f(theta):
        x=f1(theta)*f2(theta)
        return x
    return f

I=gettable('AllLeagues.xlsx',2)
U=formatdatarow(I,[[1,[2014,2015,2016,2018]],[3,['promotion','regional']]])
J=indextoteam(U,4,7)
m=len(J)
K0=formatdatacol(U,4,7,[8])
D=mu_sigma(K0)
K=normalize_data(K0,D)
sigma0=1.6#7.32
vsigma0=3
a0=3.6
va0=2
mu=0
mu0=-0.1
v=2
v0=5

n1=len(K0)
K1=[]
K10=[]
K2=[]
K20=[]
for i in range(n1):
    if i%3==0:
        K2+=[K[i]]
        K20+=[K0[i]]
    else:
        K1+=[K[i]]
        K10+=[K0[i]]

ft=gaussian_f_prior_agm(mu,mu0,sigma0,a0,v,v0,vsigma0,va0,m)
ftd=Asymetric_Gaussian_posterior(K1,m,ft)

def MHSampling(f,theta0,L,n):
    tn=np.array(theta0)
    N=len(theta0)
    L=np.array(L)
    mu=np.array([0]*N)
    cov=np.identity(N)
    A=[]
    for i in range(n):
        a=L*np.random.multivariate_normal(mu,cov) #Here I chose to use a Gaussian
        tp=tn+a
        print(tn)
        print(f(tn))
        k=f(tp)/f(tn)
        if k>=1:
            tn=tp
            A+=[tn]
        else: 
            b=np.random.uniform(0,1)
            if b<=k:
                tn=tp
            A+=[tn]
    return A

B=[0]*(m+3)
B[m]=0
B[m+1]=1.6
B[m+2]=4.8
L=MHSampling(ftd,B,[0.05]*(m+3),3000)
a=[np.float64(0)]*(m+3)
for i in range(len(L)):
    if i>=len(L)/2:  #in order to have samples that are independent from theta0. This bound can be changed
        a+=L[i]
n=len(L)/2
a=a/n
print(a)

def Asymetric_gaussian_prediction(theta_estimate,T,D):
    A=[]
    m=len(theta_estimate)
    for k in range(len(T)):
        i=T[k][0]
        j=T[k][1]
        mu0=theta_estimate[m-3]
        sigma=theta_estimate[m-2]
        a=theta_estimate[m-1]
        delta=a/((1+a**2)**(1/2))
        tij=(mu0+theta_estimate[i]+theta_estimate[j]+sigma*delta*((2/math.pi)**(1/2)))*D[1]+D[0]
        A+=[tij]
    A=np.array(A)
    return A

def Asymetric_Gaussian_MSE(theta_estimate,T,D):
    MSE=0
    m=len(theta_estimate)
    n=len(T)
    for k in range(len(T)):
        i=T[k][0]
        j=T[k][1]
        mu0=theta_estimate[m-3]
        sigma=theta_estimate[m-2]
        a=theta_estimate[m-1]
        delta=a/((1+a**2)**(1/2))
        time_estimate=(mu0+theta_estimate[i]+theta_estimate[j]+sigma*delta*((2/math.pi)**(1/2)))*D[1]+D[0]
        MSE+=(T[k][2]-time_estimate)**2
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

print(Asymetric_Gaussian_MSE(a,K10,D))
print(calculate_var_data(K10))

print(Asymetric_Gaussian_MSE(a,K20,D))
print(calculate_var_data(K20))

A=Asymetric_gaussian_prediction(a,K0,D)
print(A)
B=np.linspace(27,42,40)
X=[0]*len(B)
for i in range(len(B)-1):
    for j in range(len(A)):
        if A[j]>=B[i] and A[j]<B[i+1] and i!=len(B)-1:
            X[i]+=1
plt.plot(B,X)

