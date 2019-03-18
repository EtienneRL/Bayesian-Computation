# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:16:24 2018

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
            x=f_d_thetai*x*10 #here the factor aims to have a posterior in a good range of values (not 0)
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

def LangevinMHSampling(f,theta0,L0,n,g): #g step-size for gradient computation
    tn=np.array(theta0)
    N=len(theta0)
    mu=np.array([0]*N)
    cov=np.identity(N)
    f0=logf(f)
    A=[]
    for i in range(n):
        a=L0*np.random.multivariate_normal(mu,cov) #Here I chose to use a Gaussian
        w1=np.array((L0**2)*1/2*scipy.optimize.approx_fprime(tn,f0,g))
        tp=tn+w1+a
        w2=np.array((L0**2)*1/2*scipy.optimize.approx_fprime(tp,f0,g))
        V1=np.array(tp-tn-w1)
        V2=np.array(tn-tp-w2)
        k=f(tp)/f(tn)*math.exp(-1/(2*L0**2)*((V2*V2).sum()-(V1*V1).sum()))
        print(f(tn))
        #print(k)
        if k>=1:
            tn=tp
            A+=[tn]
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

##Plotting Accuracies for different regions and years
#M=[[[1,[2014,2016,2017,2018]],[2,['Spring']],[3,['promotion','regional']]],[[1,[2014,2016,2017,2018]],[3,['promotion','regional']]],[[1,[2014,2017,2018]],[3,['promotion','regional']]],[[1,[2014,2018]],[3,['promotion','regional']]],[[1,[2014]],[3,['promotion','regional']]],[[3,['promotion','regional']]]]
#G=[[0],[0],[0],[0]]   
#B=[[0],[0],[0],[0]]
#for i1 in range(len(M)):
#    F=[1,2,3,4]
#    for t in range(len(F)):
#        I=gettable('AllLeagues.xlsx',F[t])
#        G[t]+=[i1]
#        U=formatdatarow(I,M[i1])
#        J=indextoteam(U,4,7)
#        m=len(J)
#        K=formatdatacol(U,4,7,[5])
#        n1=len(K)
#        K1=[]
#        K2=[]
#        for i in range(n1):
#            if i%7==0:
#                K2+=[K[i]]
#            else:
#                K1+=[K[i]]
#
#        mu_prior=np.array([0]*(m+1))
#        p=6
#        cov_prior=p*np.identity(m+1)
#        ft=gaussian_f_prior(mu_prior,cov_prior)
#        ftd=logit_posterior(K1,m,ft)
#
#        L=LangevinMHSampling(ftd,[1]*(m+1),0.1,1000,0.001)
#        a=[np.float64(0)]*(m+1)
#        for i in range(len(L)):
#            if i>=len(L)/2:  #in order to have samples that are independent from theta0. This bound can be changed
#                a+=L[i]
#        n=len(L)/2
#        a=a/n
#
#        u=[]
#        for i in range(len(J)):
#            u+=[[J[i],a[i]]]
#            
#        b=logit_prediction(a,K2)
#        B[t]+=[b]
#        A=[J,a,b]
#        print(i1)
#for i2 in range(4):
#    P1=G[i2]
#    P2=B[i2]
#    N1=np.array(P1)
#    print(N1)
#    N2=np.array(P2)
#    print(N2)
#    plt.plot(N1,N2)
#plt.show()

##Plotting Comparison of LCK fitting for different training set size
F=range(2,12)
I=gettable('AllLeagues.xlsx',2)
M=[[[1,[2014,2016,2017,2018]],[2,['Spring']],[3,['Promotion','Regional']]],[[1,[2014,2016,2017,2018]],[2,['Summer']],[3,['Promotion','Regional']]],[[1,[2015,2016,2014,2018]],[2,['Spring']],[3,['Promotion','Regional']]],[[1,[2015,2016,2014,2018]],[2,['Summer']],[3,['Promotion','Regional']]],[[1,[2014,2015,2017,2018]],[2,['Spring']],[3,['Promotion','Regional']]],[[1,[2014,2015,2017,2018]],[2,['Summer']],[3,['Promotion','Regional']]]]
for t in M:
    G=[]   
    B=[]
    U=formatdatarow(I,t)
    J=indextoteam(U,4,7)
    m=len(J)
    K=formatdatacol(U,4,7,[5])
    n1=len(K)
    for i1 in range(len(F)):
        G+=[i1]
        K1=[]
        K2=[]
        for i in range(n1):
            if i%F[i1]==0:
                K2+=[K[i]]
            else:
                K1+=[K[i]]

        mu_prior=np.array([0]*(m+1))
        p=6
        cov_prior=p*np.identity(m+1)
        ft=gaussian_f_prior(mu_prior,cov_prior)
        ftd=logit_posterior(K1,m,ft)

        L=LangevinMHSampling(ftd,[1]*(m+1),0.1,1500,0.001)
        a=[np.float64(0)]*(m+1)
        for i in range(len(L)):
            if i>=len(L)/2:  #in order to have samples that are independent from theta0. This bound can be changed
                a+=L[i]
        n=len(L)/2
        a=a/n

        u=[]
        for i in range(len(J)):
            u+=[[J[i],a[i]]]
            
        b=logit_prediction(a,K2)
        B+=[b]
        A=[J,a,b]
        print(t)
    G=np.array(G)
    B=np.array(B)
    plt.plot(G,B)
plt.show()

##Plotting path of LangevinMH
#I=gettable('AllLeagues.xlsx',2)
#U=formatdatarow(I,[[1,[2014,2016,2017,2018]],[2,['Summer']],[3,['Promotion','regional']]])
#J=indextoteam(U,4,7)
#m=len(J)
#K=formatdatacol(U,4,7,[5])
#n1=len(K)
#K1=[]
#K2=[]
#for i in range(n1):
#    if i%11==0:
#        K2+=[K[i]]
#    else:
#        K1+=[K[i]]
#
#mu_prior=np.array([0]*(m+1))
#p=6
#cov_prior=p*np.identity(m+1)
#ft=gaussian_f_prior(mu_prior,cov_prior)
#ftd=logit_posterior(K1,m,ft)
#
#L=LangevinMHSampling(ftd,[0]*(m+1),0.1,3000,0.001)
#a=[np.float64(0)]*(m+1)
#for i in range(len(L)):
#    if i>=len(L)/2:  #in order to have samples that are independent from theta0. This bound can be changed
#        a+=L[i]
#n=len(L)/2
#a=a/n
#print(a)
#
###Plotting epochs of MH for 2 dimensions of theta
#Q=[4,5]
#S1=[]
#S2=[]
#for i in range(len(L)):
#    pt1=Q[0]
#    pt2=Q[1]
#    Li=L[i]
#    S1+=[Li[pt1]]
#    S2+=[Li[pt2]]
#S1=np.array(S1)
#S2=np.array(S2)
#plt.plot(S1,S2)
#print(a[4],a[5])
#
#u=[]
#for i in range(len(J)):
#    u+=[[J[i],a[i]]]
#            
#b=logit_prediction(a,K2)
#print(b)
#A=[J,a,b]
#print(u)
#print(a)

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
#    L=LangevinMHSampling(ftd,[0]*(m+1),0.1,3000,0.0001)
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

##Code for random training set
#        n1=int(len(K)/2)
#        K2=[]
#        for i in range(n1):
#            v=np.random.choice(range(len(K)))
#            v0=K[v]
#            K2+=[v0]
#            K.remove(v0)
#        K1=K