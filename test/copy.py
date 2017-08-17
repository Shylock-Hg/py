# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 09:38:47 2017

@author: sily-
"""

import copy

l1 = ["str1",3,["st",2,"mbed"],(1,"str"),{1,3,"str"},{"key1":1,"key2":3}]

l2 = copy.copy(l1)

l3 = copy.deepcopy(l1)

print(id(l1),[id(ele) for ele in l1])
print(id(l2),[id(ele) for ele in l2])
print(id(l3),[id(ele) for ele in l3])

l4 = l1[:]
print(l4 is l1,[l4[i] is l1[i] for i in range(len(l1))])
