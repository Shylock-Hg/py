'''
@brief: get started to pylab
@author: TcAth2s
@email: tacth2s@icloud.com
@time: 2017/8/8
'''

import pylab

#coordinate list
x = [1,2,3,4]
y = [5,2,9,4]

pylab.figure(1)
pylab.plot(x,y)


#plot the gain for 20-years invest in 5% interest rate
printcepal = 10000
gain = [printcepal*1.05**i for i in range(20)]

pylab.figure(2)
pylab.title('gain for invest in 20 years in 5% interest rate')
pylab.xlabel('year')
pylab.ylabel('gain')
pylab.plot(gain)
pylab.show()
