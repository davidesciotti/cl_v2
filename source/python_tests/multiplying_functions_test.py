from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
#from sympy import *
from functools import partial



def f(x):
    return x**2

def g(x,y):
    return 3*y+x

def multiply(x,y):
    return f(x)*g(x,y)


# create a new function that multiplies by 2
#multiply_partial = partial(multiply, x)
#def integrand(x,y):
#    return x*np.exp(x/y)
#
#vec_int = np.vectorize(integrand)
#y = np.linspace(0, 10, 100)
#vec_int(y)


def h(y):
    #integro in dx
    result = quad(multiply(x,y), 2, 4)
    return result

y = np.linspace(1,10)
h(y)

#x = Symbol('x')
#y = Symbol('y')
#
#f = x*y + x**2
#a = 1
#b = 2
#
#h = integrate(f, (x, a, b))

#print(multiply(1,2))
#
#y = np.linspace(0,10)
#x = np.linspace(0,10)
#
#print(h(y))
##plt.plot(y, integrate(x, y))

