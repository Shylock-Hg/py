'''
@brief: draw the tempreture figure by the file julyTemps.txt
@author: TcAth2s
@email: tcath2s@icloud.com
@time:2017/8/8
'''

import pylab

#read file
f = open('julyTemps.txt','r')
lines = f.readlines();
del lines[:6]

strs = []
for line in lines:
    strs.append(line.split(' '))

tp_high = []
tp_low = []
for s in strs:
    tp_high.append(int(s[1]))
    tp_low.append(int(s[2]))

#plot
pylab.figure(1)
pylab.title('the high & low tempreture in these days')
pylab.xlabel('days')
pylab.ylabel('tempreture')
pylab.plot(tp_high)
pylab.plot(tp_low)
pylab.show()
