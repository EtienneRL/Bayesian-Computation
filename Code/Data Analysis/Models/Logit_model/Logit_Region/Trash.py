# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 20:06:27 2018

@author: étienne
"""
from numpy import random
A=[[0,1,2],[3,4,5],[-3,5,6]]
a=random.choice(range(len(A)))
print(a)
A.remove(A[a])
print(A)