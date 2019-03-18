# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:47:18 2018

@author: étienne
"""
import xlrd

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
A2=formatdatastrictrowin(A1,[[0,['EULCS','NALCS','LMS','LCK']],[1,[2015]],[2,['Spring']]]) #important that strictrowin here
A3=indextoregion(A2,0)
A4=fusionTindexregion(A2,A3)
A5=indextoteam(A2,4,7)
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
print(A9)