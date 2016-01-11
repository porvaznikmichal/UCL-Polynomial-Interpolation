import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import trapz

def weights(x):
    """Computes barycentric weights for supplied interpolation points.

Parameters
----------
x: array with n distinct real entries
   Interpolation points.

Returns
-------
w: array with n real entries
   Barycentric weights.

"""
    n = x.shape[0]
    w = np.ones(n)

    for j in range(1,n):
        for k in range(j):
            w[k] *= (x[k] - x[j])
            w[j] *= (x[j] - x[k])

    for j in range(n):
        w[j] = 1/w[j]

    return w

# The following procedure interp() is the most important part of code in this
# file on which all the subsequent functions rely to compute the interpolant.

def interp(x, f, points=1000, cheb=False):
    """Computes barycentric interpolant for supplied interpolation data.

Parameters
----------
x: array with n distinct real entries
   Interpolation points.
   
f: array with n real entries
   Functional values at interpolation points.
   
points: positive integer, optional, if not supplied then points=1000
        Number of points for interpolant to be evaluated at.
        
cheb: boolean, optional, if not supplied then cheb=Flase
      If True then uses pre specified Barycentric weights,
      otherwise uses weights(x).   

Returns
-------
x1: array with points elements
    Evaluation points of interpolant.

p: array with points elements
   Interpolant values at evaluation points.

"""
    n = x.shape[0]
    (a,b) = (min(x[j] for j in range(n)), max(x[j] for j in range(n)))  
    x1 = np.linspace(a,b,points)
    numer = np.zeros(points)
    denom = np.zeros(points)
    p = np.zeros(points)

    # The following if statement simplifies the assignment of weights
    # for the case of Chebyshev points   
    if cheb:
        w = np.ones(n)
        for i in range(1, n-1):
            w[i] *= (-1)**i
        w[0] = 0.5
        w[n-1] = 0.5
    else:
        w = weights(x)

    for i in range(n):
        xdiff = x1 - x[i]
        # The following if statement eliminates division by zero.
        if 0 in xdiff:
            for j in range(points):
                if xdiff[j] == 0:
                    xdiff[j] = 1
                    break                
        div = w[i] / xdiff
        numer += div*f[i]
        denom += div
            
    p = numer /  denom

    # The following corrects values at points which were changed to eliminate
    # division by zero.
    for i in range(n):
        for j in range(points):
            if x[i] == x1[j]:
                p[j] = f[i]

    return (x1, p)

def runge_1(n):
    """Plots function f = 1/(1 + x**2) and its interpolant of degree n+1,
interpolated at equidistant points, on interval [-5,5] to demonstrate the
Runge phenomenon.

Paramters
---------
n: integer
   Determines the number of interpolation points.

"""

    x = np.linspace(-5, 5, n)
    f = 1 / (1 + x**2)
    (x1, p) = interp(x, f, points=max(2*n, 1000))
    f1 = 1/ (1 + x1**2)

    plt.ylabel('y')
    plt.xlabel('x')
    plt.yscale('linear')
    plt.plot(x1, f1, x1, p, '--')

def runge_2(n):
    """Plots function f(x) = cos(x) and its interpolant of degree n+1,
interpolated at equidistant points, on interval [-5,5] to examine the
Runge phenomenon.

Paramters
---------
n: integer
   Determines the number of interpolation points.

"""

    x = np.linspace(-5, 5, n)
    f = np.cos(x)
    (x1, p) = interp(x, f, points=max(2*n, 1000))
    f1 = np.cos(x1)

    plt.ylabel('y')
    plt.xlabel('x')
    plt.yscale('linear')
    plt.plot(x1, f1, x1, p, '--')

def cheb(n):
    """Plots function f(x) = 1/(1 + x**2) and its interpolant of degree n+1,
interpolated at Chebychev points, on interval [-5,5] to demonstrate the
elimination of Runge phenomenon by appropriate choice of points.

Paramters
---------
n: integer
   Determines the number of interpolation points.

"""

    x = np.zeros(n)

    for i in range(n):
        x[i]= 5 * np.cos((np.pi * i) / (n-1))

    f = 1 / (1 + x**2)
    (x1,p) = interp(x, f, points=max(2*n, 1000),cheb=True)
    f1 = 1/ (1 + x1**2)

    plt.ylabel('y')
    plt.xlabel('x')
    plt.yscale('linear')
    plt.plot(x1, f1, x1, p, '--')

def f(n):
    """Plots function f(x) = sqrt(abs(x)) and its interpolant of degree n+1,
interpolated at Chebychev points, on interval [-1,1].

Paramters
---------
n: integer
   Determines the number of interpolation points.

"""

    x = np.zeros(n)

    for i in range(n):
        x[i]= np.cos((np.pi * i) / (n-1))

    f = np.sqrt(abs(x))
    (x1, p) = interp(x, f, points=max(2 * n + 1, 1001), cheb=True)
    f1 = np.sqrt(abs(x1))

    plt.ylabel('y')
    plt.xlabel('x')
    plt.yscale('linear')
    plt.plot(x1, p, '-')

def sign(x):
    """Produces vector with sign of every element

Parameters
----------
x: array with n real entries

Returns
-------
w: array with n real entries

"""

    n = x.shape[0]
    w = np.ones(n)

    for i in range(n):
        if x[i] < 0:
            w[i] = -1

    return w

def g(n):
    """Plots function f(x) = sign(x) (i.e. f(x) = -1 if x < 0 and f(x) =1
otherwise) and its interpolant of degree n+1,
interpolated at Chebychev points, on interval [-1,1].

Paramters
---------
n: integer
   Determines the number of interpolation points.

"""

    x = np.zeros(n)

    for i in range(n):
        x[i]= np.cos((np.pi * i) / (n-1))

    f = sign(x)
    (x1, p) = interp(x, f, points=max(2*n, 1000), cheb=True)
    f1 = sign(x1)

    plt.ylabel('y')
    plt.xlabel('x')
    plt.yscale('linear')
    plt.plot(x1, p, '-')


# The following four functions conv1(), conv2(), conv3() and conv4() are all of
# the same form and all plot some form of error between functions and
# their interpolants.

def conv1(n, m, k=1):
    """Plots maximum error between function f(x) = 1/(1 + x**2) and its
interpolant in Chebyshev points on interval [-1,1] against the degree of
interpolant.

Parameters
----------
n: positive integer
   Starting point (inclusive).

m: positive integer
   End point (excusive).

k: positive integer, optional, if not supplied then k=1
   Step.

"""

    plt.ylabel('max|f - p|')
    plt.xlabel('n')
    plt.yscale('log')
    
    for i in range(n, m, k):
        x = np.zeros(i)

        for j in range(i):
            x[j]= np.cos((np.pi * j) / (i-1))

        f = 1 / (1 + 16*x**2)
        (x1, p) = interp(x, f, cheb=True)
        f1 = 1/ (1 + 16*x1**2)

        mdiff = max(abs(f1[j] - p[j]) for j in range(1000))

        plt.plot(i, mdiff, 'ko')

def conv2(n, m, k=1):
    """Plots maximum error between function f(x) = cos(x) and its
interpolant in Chebyshev points on interval [-1,1] against the degree of
interpolant.

Parameters
----------
n: positive integer
   Starting point (inclusive).

m: positive integer
   End point (excusive).

k: positive integer, optional, if not supplied then k=1
   Step.
"""
    plt.ylabel('max|f - p|')
    plt.xlabel('n')
    plt.yscale('log') 
    for i in range(n, m, k):
        x = np.zeros(i)

        for j in range(i):
            x[j]= np.cos((np.pi * j) / (i-1))

        f = np.cos(x)
        (x1, p) = interp(x, f, cheb=True)
        f1 = np.cos(x1)

        mdiff = max(abs(f1[j] - p[j]) for j in range(1000))

        plt.plot(i, mdiff, 'k*')

def conv3(n, m, k=1):
    """Plots maximum error between function f(x) = sqrt(abs(x)) and its
interpolant in Chebyshev points on interval [-1,1] against the degree of
interpolant.

Parameters
----------
n: positive integer
   Starting point (inclusive).

m: positive integer
   End point (excusive).

k: positive integer, optional, if not supplied then k=1
   Step.

"""

    plt.ylabel('max|f - p|')
    plt.xlabel('n')
    plt.yscale('log')
    
    for i in range(n, m, k):
        x = np.zeros(i)

        for j in range(i):
            x[j]= np.cos((np.pi * j) / (i-1))

        f = np.sqrt(abs(x))
        (x1, p) = interp(x, f, cheb=True)
        f1 = np.sqrt(abs(x1))

        mdiff = max(abs(f1[j] - p[j]) for j in range(1000))

        plt.plot(i, mdiff,'ko')


def conv4(n, m, k=1):
    """Plots integral of abs(f(x)-p(x))dx from x = -1 to x = 1, where
f(x) = sign(x) and p is its interpolant in Chebyshev points, against the degree
of interpolant.

Parameters
----------
n: positive integer
   Starting point (inclusive).

m: positive integer
   End point (excusive).

k: positive integer, optional, if not supplied then k=1
   Step.

"""

    plt.ylabel('integral(abs(f(x) - p(x))dx')
    plt.xlabel('n')
    plt.yscale('log')
    
    for i in range(n, m, k):
        x = np.zeros(i)

        for j in range(i):
            x[j]= np.cos((np.pi * j) / (i-1))

        f = sign(x)
        (x1, p) = interp(x, f, cheb=True)
        f1 = sign(x1)

        mdiff = trapz(abs(f1 - p), x1)

        plt.plot(i, mdiff, 'k*')
    
    
