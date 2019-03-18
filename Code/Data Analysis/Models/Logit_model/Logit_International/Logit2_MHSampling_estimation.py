# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:39:12 2018

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
#Goal: get all distinct teams of table and their identifier

def indextoregion(T,o): #T is modified table: gettable without international events #o column no of regions 
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
    return B
#len(B)=number of regions
#Goal: get all regions and their identifier with a non modified table

def formatdatarowex(T,excluderow): #excluderowe: exclude row j for i in excluderow[0] T[i][j] is in excluderow[1]
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
#Goal: take basis table and exclude rows containing banned words

def formatdatarowin(T,includerow): #excluderowe: exclude row j for i=excluderow[0] T[i][j] is in excluderow[1]
    n=len(T)
    m=len(includerow)
    T1=[]
    for i in range(n):
        a=0
        for j in range(m):
            e=includerow[j]
            for k in range(len(e[1])):
                if T[i][e[0]]==e[1][k]:
                    a=1
        if a==1:
            T1+=[T[i]]
    return T1
#Goal: take basis table, include rows that contain at least one given word

def formatdatastrictrowin(T,includerow):
    n=len(T)
    m=len(includerow)
    T1=[]
    for i in range(n):
        b=0
        for j in range(m):
            e=includerow[j]
            a=0
            for k in range(len(e[1])):
                if T[i][e[0]]==e[1][k]:
                    a=1
            if a==0:
                b=1
        if b==0:
            T1+=[T[i]]
    return T1
#Goal: take basis table, include rows that contain at least one given word

def fusionTindexregion(T1,T2):#T1 big table #T2 region table
    for i in range(len(T1)):
        j=0
        a=0
        while a==0:
            if T1[i][0]==T2[j][0]:
                T1[i]+=[T2[j][1]]
                a=1
            j+=1
    return T1

def fusionTeamRegion(T1,T2): #T1 big table #T2 team table
    for i in range(len(T2)):
        j=0
        a=0
        while a==0:
            if T1[j][4]==T2[i][0] or T1[j][7]==T2[i][0]:
                T2[i]+=[T1[j][-1]]
                a=1
            j+=1
    return T2

A1=gettable('AllLeagues.xlsx',0)
A=formatdatarowex(A1,[[3,['Promotion']]])
A2=formatdatastrictrowin(A,[[0,['EULCS','NALCS','LMS','LCK']],[1,[2015]]]) #important that strictrowin here
A3=indextoregion(A2,0)
m2=len(A3)
A4=fusionTindexregion(A2,A3)
A5=indextoteam(A2,4,7)
m1=len(A5)
A6=fusionTeamRegion(A4,A5)
print(A6)

def formatdatacolregions(T1,T2,a,b,col): #T2 is modified indextoteam
    n=len(T1)
    A=[]
    for i in range(n):
        K=[0,0,0,0]
        a1=0
        a2=0
        j=0
        while (a1==0 or a2==0):
            if T2[j][0]==T1[i][a]:
                a1=1
                K[0]=T2[j][1]
                K[2]=T2[j][2]
            if T2[j][0]==T1[i][b]:
                a2=1
                K[1]=T2[j][1]
                K[3]=T2[j][2]
            j+=1
        for e in range(len(col)):
            K+=[T1[i][col[e]]]
        A+=[K]
    return A

def onlytaketeams(T):
    A=[]
    for i in range(len(T)):
        A+=[T[i][0]]
    return A

A7=onlytaketeams(A6)
A8=formatdatastrictrowin(A1,[[4,A7],[7,A7]])
A9=formatdatacolregions(A8,A6,4,7,[5])

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
            x=f_d_thetai*x*1.9 #here the factor aims to have a posterior in a good range of values (not 0) we change it in function of the number of iterations
        return x
    return f_d_theta

def gaussian_f_prior(mu,cov):
    def k(theta):
        b=scipy.stats.multivariate_normal(mu,cov)
        a=b.pdf(theta)
        return a
    return k

def logit2_posterior(D,m1,m2,prior):#m number of teams for logit_parametric_model
    f1=logit2_parametric_model(D,m1,m2)
    f2=prior
    def f(theta):
        x=f1(theta)*f2(theta)
        return x
    return f

mu_prior=[0]*(m1+m2+1)
p=2
cov_prior=p*np.identity(m1+m2+1)
ft=gaussian_f_prior(mu_prior,cov_prior)
ftd=logit2_posterior(A9,m1,m2,ft)

n1=len(A9)
K1=[]
K2=[]
for i in range(n1):
    if i%10==0:
        K2+=[A9[i]]
    else:
        K1+=[A9[i]]

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
        print(f(tn))
        print(tn)
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

L=MHSampling(ftd,[0]*(m1+m2+1),[0.1]*(m1+m2+1),1500)
a=[np.float64(0)]*(m1+m2+1)
for i in range(len(L)):
    if i>=len(L)/2:  #in order to have samples that are independent from theta0. This bound can be changed
        a+=L[i]
n=len(L)
a=a/n

def logit_prediction(theta_estimate,m1,m2,T):
    A=[]
    for k in range(len(T)):
        i1=T[k][0]
        j1=T[k][1]
        i2=T[k][2]
        j2=T[k][3]
        pij=theta_estimate[i1]-theta_estimate[j1]+theta_estimate[m1+i2]-theta_estimate[m1+j2]+theta_estimate[m1+m2-1]
        c=0
        if pij>=0:
            c=1
        if c==T[k][-1]:
            A+=[1]
        else:
            A+=[0]
    A=np.array(A)
    c=np.sum(A)/len(A)*100
    return c

b=logit_prediction(a,m1,m2,K2)
print(b)

for k in range(len(A3)):
    A3[k]+=[a[m1+A3[k][1]]]
print(A3)

for k in range(len(A6)):
    A6[k]+=[a[A6[k][1]]+a[m1+A6[k][2]]]
print(A6)
