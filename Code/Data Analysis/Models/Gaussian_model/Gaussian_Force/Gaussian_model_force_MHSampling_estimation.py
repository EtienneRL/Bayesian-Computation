# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:58:09 2018

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

I=gettable('AllLeagues.xlsx',2)
U=formatdatarow(I,[[3,['promotion','regional']]])
J=indextoteam(U,4,7)
m=len(J)
K=formatdatacol(U,4,7,[5,8])
n1=int(len(K)/2)
n1=len(K)
K1=[]
K2=[]
for i in range(n1):
    if i%4==0:
        K2+=[K[i]]
    else:
        K1+=[K[i]]
#mu_prior=[0]*m
#d=-5
#mu_prior=mu_prior+[d]  #(m+1) parameters: force of each team and blue team bias
mu_prior=np.array([0]*(m+1))
p=6
cov_prior=p*np.identity(m+1)
ft_force=gaussian_f_prior(mu_prior,cov_prior)
ftd_force=logit_posterior(K1,m,ft_force)
#ok we got the posterior now
#now let's sample using Metropolis-Hastings

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

L_force=MHSampling(ftd_force,[1]*(m+1),[0.1]*(m+1),3000)
a=[np.float64(0)]*(m+1)
for i in range(len(L_force)):
    if i>=len(L_force)/2:  #in order to have samples that are independent from theta0. This bound can be changed
        a+=L_force[i]
n=len(L_force)/2
a_force=a/n
print(a_force)

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
print(b)
#Now that we've computed an estimator of the force, we can build another one for game duration

def Gaussian_force_parametric_model2(T,force_estimate,m):
    def f_d_theta(theta):
        x=1
        mu0=theta[m]
        alpha=theta[m+2]
        sigma=theta[m+1]
        for k in range(len(T)):
            i=T[k][0]
            j=T[k][1]
            fi=(1/(2*math.pi*(sigma**2)))**(1/2)*math.exp(-1/2*(((T[k][3]-(mu0+theta[i]+theta[j]+alpha*(abs(force_estimate[i]-force_estimate[j]+force_estimate[m]))))/sigma)**2))*50
            x=fi*x
        return x
    return f_d_theta

def gaussian_force_f_prior_gm(mu,mu0,sigma0,mu1,v,v0,vsigma0,v1,m):
    mean=[0]*(m+3)
    for i in range(m):
        mean[i]=mu
    mean[m]=mu0
    mean[m+1]=sigma0
    mean[m+2]=mu1
    var=v*np.identity(m+3)
    var[m][m]=v0
    var[m+1][m+1]=vsigma0
    var[m+2][m+2]=v1
    def k(theta):
        b=scipy.stats.multivariate_normal(mean,var)
        a=b.pdf(theta)
        return a
    return k

def Gaussian_force_posterior(D,force_estimate,m,prior):#m number of teams for logit_parametric_model AND NOT number of parameters
    f1=Gaussian_force_parametric_model2(D,force_estimate,m)
    f2=prior
    def f(theta):
        x=f1(theta)*f2(theta)
        return x
    return f

#mu=0
#mu0=38.13162063
#sigma0=6
#mu1=-0.64033355
#v=2
#v0=5
#vsigma0=3
#v1=1
#
#ft_time=gaussian_force_f_prior_gm(mu,mu0,sigma0,mu1,v,v0,vsigma0,v1,m)
#ftd_time=Gaussian_force_posterior(K1,a_force,m,ft_time)
#
#B=[0]*(m+3)
#B[m]=35
#B[m+1]=5
#B[m+2]=-1
#L_time=MHSampling(ftd_time,B,[0.1]*(m+3),15000)
#a=[np.float64(0)]*(m+3)
#for i in range(len(L_time)):
#    if i>=len(L_time)/2:  #in order to have samples that are independent from theta0. This bound can be changed
#        a+=L_time[i]
#n=len(L_time)/2
#a_time=a/n
#a_time_MLE=L_time[-1]
#print(a_time)

def gaussian_force_prediction2(theta_estimate,force_estimate,T):
    A=[]
    m=len(theta_estimate)
    for k in range(len(T)):
        i=T[k][0]
        j=T[k][1]
        tij=theta_estimate[m-3]+theta_estimate[i]+theta_estimate[j]+theta_estimate[m-1]*abs(force_estimate[i]-force_estimate[j]+force_estimate[m-3])
        A+=[tij]
    A=np.array(A)
    return A

def Gaussian_force_MSE(theta_estimate,force_estimate,T):
    MSE=0
    m=len(theta_estimate)
    n=len(T)
    for k in range(len(T)):
        i=T[k][0]
        j=T[k][1]
        MSE+=(T[k][3]-(theta_estimate[m-3]+theta_estimate[i]+theta_estimate[j]+theta_estimate[m-1]*abs(force_estimate[i]-force_estimate[j]+force_estimate[m-3])))**2
    MSE=MSE/n
    return MSE

def calculate_var_data(T):
    n=len(T)
    m=0
    for i in range(n):
        m+=T[i][3]
    m=m/n
    v=0
    for i in range(n):
        v+=(T[i][3]-m)**2
    v=v/n
    return v

#print(Gaussian_force_MSE(a_time,a_force,K2))
#print(calculate_var_data(K2))

K3=[]
for k in range(len(K2)):
    i=K2[k][0]
    j=K2[k][1]
    pij=a_force[i]-a_force[j]+a_force[m]
    if pij>=0:
        c=1
    else:
        c=0
    if c==K2[k][2]:
        K3+=[K2[k]]

#print(Gaussian_force_MSE(a_time,a_force,K3))
#print(calculate_var_data(K3))

def plt_distribution(T,a0,b0,n):
    x=np.linspace(a0,b0,n)
    y=[0]*(n)
    for i in range(len(T)):
        for j in range(len(x)):
            v=T[i]
            if v<x[0]:
                y[0]+=1
            elif v>=x[-1]:
                y[-1]+=1
            elif v>=x[j] and v<x[j+1]:
                y[j]+=1  
    y=np.array(y)
    plt.plot(x,y)
    
def plt_distribution2(T,a0,b0,n,j0):
    x=np.linspace(a0,b0,n)
    y=[0]*(n)
    for i in range(len(T)):
        for j in range(len(x)):
            v=T[i][j0]
            if v<x[0]:
                y[0]+=1
            elif v>=x[-1]:
                y[-1]+=1
            elif v>=x[j] and v<x[j+1]:
                y[j]+=1  
    y=np.array(y)
    plt.plot(x,y)

#plt_distribution(gaussian_force_prediction2(a_time,a_force,K2),20,90,55)
#plt_distribution2(K2,20,90,55,3)

#91.26698318626875 (good force predictions)
#79.10743801652895
#59.432444091333394
#51.68975069252078
#
#55.24074713510472 (good force predictions)
#56.930565289256236
#47.72249115902358
#47.37565760848075

#x=[]
#y=[]
#for i in range(len(K)):
#    k1=K[i][0]
#    k2=K[i][1]
#    n1=np.random.normal(0,0.3)
#    n2=np.random.normal(0,0.3)
#    f1=a_force[k1]+n1
#    f2=a_force[k2]+n2
#    x+=[f1,f2]
#    y+=[K[i][3],K[i][3]]
#plt.plot(x,y,'ro')
#plt.show()

x=[]
y=[]
for i in range(len(K)):
    k1=K[i][0]
    k2=K[i][1]
    n1=np.random.normal(0,0.5)
    f1=abs(a_force[k1]-a_force[k2])+n1
    x+=[f1]
    y+=[K[i][3]]
#plt.plot(x,y,'ro')
xprim=np.linspace(-0.5,4,70)
yprim=[0]*(70)
yprimprim=[0]*70
for i in range(len(y)):
    for j in range(len(xprim)):
        v=x[i]
        if v<xprim[0]:
            yprim[0]+=1
        elif v>=xprim[-1]:
            yprim[-1]+=1
        elif v>=xprim[j] and v<xprim[j+1]:
            yprim[j]+=1 
            yprimprim[j]+=y[i]
for j in range(len(yprimprim)):
    if yprimprim[j]!=0:
        yprimprim[j]=yprimprim[j]/yprim[j]
yprim=np.array(yprimprim)
plt.plot(xprim,yprimprim)
plt.show()