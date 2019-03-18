# -*- coding: utf-8 -*-
"""
Created on Mon May 21 08:47:51 2018

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
        k=f(tp)/f(tn)
        if k>=1:
            tn=tp
            A+=[tn]
            print(tn)
        else: 
            b=np.random.uniform(0,1)
            if b<=k:
                tn=tp
            A+=[tn]
    return A

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

##Plotting results for each split
#I=gettable('AllLeagues.xlsx',2)
#M=[[[1,[2014,2016,2017,2018]],[2,['Spring']],[3,['promotion','regional']]],[[1,[2014,2016,2017,2018]],[2,['Summer']],[3,['promotion','regional']]],[[1,[2014,2015,2017,2018]],[2,['Spring']],[3,['promotion','regional']]],[[1,[2014,2015,2017,2018]],[2,['Summer']],[3,['promotion','regional']]],[[1,[2014,2016,2015,2018]],[2,['Spring']],[3,['promotion','regional']]],[[1,[2014,2016,2015,2018]],[2,['Summer']],[3,['promotion','regional']]]]
#X1=range(len(M))
#X2=[]
#for i1 in range(len(M)):
#    U=formatdatarow(I,M[i1])
#    J=indextoteam(U,4,7)
#    m=len(J)
#    K=formatdatacol(U,4,7,[5])
#    
#    n1=len(K)
#    K1=[]
#    K2=[]
#    for i in range(n1):
#        if i%10==0:
#            K2+=[K[i]]
#        else:
#            K1+=[K[i]]
#    #mu_prior=[0]*m
#    #d=-5
#    #mu_prior=mu_prior+[d]  #(m+1) parameters: force of each team and blue team bias
#    mu_prior=np.array([0]*(m+1))
#    p=6
#    cov_prior=p*np.identity(m+1)
#    ft=gaussian_f_prior(mu_prior,cov_prior)
#    ftd=logit_posterior(K1,m,ft)
#    #ok we got the posterior now
#    #now let's sample using Metropolis-Hastings
#    
#    L=MHSampling(ftd,[10]*(m+1),[0.1]*(m+1),10000)
#    a=[np.float64(0)]*(m+1)
#    for i in range(len(L)):
#        if i>=len(L)/2:  #in order to have samples that are independent from theta0. This bound can be changed
#            a+=L[i]
#    n=len(L)/2
#    a=a/n
#    print(a)
#    
#    b=logit_prediction(a,K2)
#    X2+=[b]
#    print(b)
#    print(len(K))
#plt.plot(X1,X2)
#plt.show()

#[ 0.32326333 -0.20067234 -0.06169991  1.33031058 -0.75387582  1.63940812  0.73089841  3.14665047  0.18032729 -0.143077   -1.69755074 -1.16661391  0.11602202]

I=gettable('AllLeagues.xlsx',2)
U=formatdatarow(I,[[1,[2014,2016,2017,2018]],[2,['Summer']],[3,['promotion','regional']]])
J=indextoteam(U,4,7)
m=len(J)
K=formatdatacol(U,4,7,[5])
n1=len(K)
K1=[]
K2=[]
for i in range(n1):
    if i%11==0:
        K2+=[K[i]]
    else:
        K1+=[K[i]]

mu_prior=np.array([0]*(m+1))
p=6
cov_prior=p*np.identity(m+1)
ft=gaussian_f_prior(mu_prior,cov_prior)
ftd=logit_posterior(K1,m,ft)

L=MHSampling(ftd,[10]*(m+1),0.2,5000)
a=[np.float64(0)]*(m+1)
for i in range(len(L)):
    if i>=len(L)/2:  #in order to have samples that are independent from theta0. This bound can be changed
        a+=L[i]
n=len(L)/2
a=a/n
print(a)

##Plotting iterations for 2 dimenstions of theta
Q=[4,5]
S1=[]
S2=[]
for i in range(len(L)):
    pt1=Q[0]
    pt2=Q[1]
    Li=L[i]
    S1+=[Li[pt1]]
    S2+=[Li[pt2]]
S1=np.array(S1)
S2=np.array(S2)
plt.plot(S1,S2)