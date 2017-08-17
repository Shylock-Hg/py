# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 09:16:52 2017

@author: sily-
"""

#number
n = 1
print(id(n))
n = 2
print(id(n))

#str
s = "java"
print(id(s))
s = "python"
print(id(s))

#turple
t = (1,2,3)
print(id(t))
t = (4,5,6)
print(id(t))

#list
l = [1,"list",3,[1,2,3]]
print(id(l))
print([id(ele) for ele in l])
l = [2,"list3",1,[6,4,2]]
print(id(l))
print([id(ele) for ele in l])

#set
se = {1,3,4}
print(id(se))
se = {4,7,8}
print(id(se))

#dict
d = {"key1":1,"key2":3}
print(id(d))
d = {"key1":4,"key2":7}
print(id(d))




n2 = 5
n2 = n

s2 = "hello"
s2 = s

t2 = (3,7,9)
t2 = t

l2 = [2,"str",7,[7,9,11]]
l2 = l

se2 = {4,5,7}
se2 = se

d2 = {"ke":5,"key4":0}
d2 = d

print(n2 is n,s2 is s,t2 is t,l2 is l,se2 is se,d2 is d)

print(l2)
print([l[i] is l2[i] for i in range(4)])


print(id(l))
l.append(3)
print(id(l))

u1 = 100
u2 = u1+123
print(u1 is u2)