# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:06:10 2017

@author: sily-
"""

def func(*vars):
    print(type(vars))
    
def func2(**vars):
    print(type(vars))

func(1,2,3)

func2(key1 = 1,key2 = 2)

import sys
print(sys.path)