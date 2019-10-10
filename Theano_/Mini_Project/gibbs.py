from sys import argv
from math import *
from random import random
import numpy as np
import matplotlib.pyplot as plt

def genexp(lamb):
  return (-1.0/lamb)*log(random())

def gennor(mu,sigma):
  theta=random()*2*pi
  rsq=genexp(0.5)
  z=sqrt(rsq)*cos(theta)
  return mu+z*sigma

n=eval(argv[1])
rho=eval(argv[2])
x=0
y=0
xa = np.zeros((n))
ya = np.zeros((n))
sig=sqrt(1-rho*rho)
for i in range(n):
  x=gennor(rho*y,sig)
  y=gennor(rho*x,sig)
  xa[i]=x
  ya[i]=y
  #print x,y
temp = xa*ya
plt.figure(1)
plt.plot(xa[0:20],'+')
plt.plot(ya[0:20],'.')
plt.show()
temp =np.sum(temp)/n
print(temp)
print(sig)
plt.figure(2)
plt.hist(xa, bins='auto')
plt.title("Histogram with 'auto' bins")
plt.show()