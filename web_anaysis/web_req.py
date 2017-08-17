# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:33:49 2017

@author: sily-
"""

import requests as rq
import bs4

hfile = rq.get('https://en.wikipedia.org/wiki/Dead_Parrot_sketch')

#print(hfile.text)
print(type(hfile.text))

bsf = bs4.BeautifulSoup(hfile.text,'html.parser')
atags = bsf.p.find_all('a')
print(atags)
print(type(atags))
