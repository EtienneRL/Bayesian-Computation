# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:17:06 2018

@author: étienne
"""
import xlrd
wb=xlrd.open_workbook('EULCS.xlsx')
s=wb.sheet_names()
print(s)
y=wb.sheet_by_index(0)
z1=y.row(0)
print(z1)
z2=y.ncols
print(z2)
z3=y.nrows
print(z3)
A=[]
for i in range(z3):
    A+=[y.row(i)]
print(A)

def formatdata(T,indcol): #T is table from gettable #indcol is a LIST of indexes of columns to add
    n=len(T)
    A=[]
    B=[[T[0][4],1]]
    c=1
    for i in range(n):
        K=[0,0]
        t1=T[i][4]
        t2=T[i][7]
        j=0
        a1=0
        a2=0
        while j!=len(B) and (a1==0 or a2==0):
            if t1==B[j][0]:
                K[0]=B[j][1]
                a1=1
            if t2==B[j][0]:
                K[1]=B[j][1]
                a2=1
            j+=1
        if a1==0:
            c+=1
            B+=[[T[j][4],c]]
            K[0]=c
        if a2==0:
            c+=1
            B+=[[T[j][7],c]]
            K[1]=c
        for u in range(len(indcol)):
            K+=[T[i][indcol[u]]]
        print(B)
        A+=[K]
    return A #A is a list here, not an array


