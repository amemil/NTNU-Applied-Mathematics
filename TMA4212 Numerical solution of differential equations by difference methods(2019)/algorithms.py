#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:08:54 2019

@author: emilam
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from tqdm import tqdm
from scipy.integrate import solve_ivp
import scipy


def BBsolution(x,t,m,gam = 0.4,n = 1):
    """
    Exact barenblatt solution for a given point 
    """
    alp = n/(n*(m-1)+2)
    bet = alp/n
    return max(0,t**(-alp)*(gam - (bet*(m-1)*abs(x)**2)/(2*m*t**(2*bet)))**(1/(m-1)))


def refsolutionBB(m,M,N,xstart,xstop,T,lam = 0.4, n=1):
    """
    Exact Barenblatt solution
    """
    x = np.linspace(xstart,xstop,M+1)
    t = np.linspace(0.1,T,N+1)
    U2 = np.zeros((M+1,N+1))
    for i in range(M+1):
        for j in range(N+1):
            U2[i,j] = BBsolution(x[i],t[j],m)
    return x,t,U2
    

def FTCS(f,M,N,T,xstart,xstop,s,g0,g1,init = 'f'):
    """
    Forward in time, central in space
    """
    U = np.zeros((M+1,N+1))
    x = np.linspace(xstart,xstop,M+1)
    t = np.linspace(0.1,T,N+1)
    if init == 'BB': #INITIALIZING WITH BARENBLATT
        _,_,Ut = refsolutionBB(s,M,N,xstart,xstop,T)
        U[:,0] = Ut[:,0]
        
    else: #INITIALIZING WITH e-(x^2)
        U[:,0] = f(x)
    
    #BCs
    U[0,:], U[-1,:] = g0(t),g1(t)
    
    
    h = (x[-1]-x[0])/M    # Stepsize in space
    k = T/N               # Stepsize in time

    r = k/(h**2)
    print('r = {:},h = {:}'.format(r,h))
    
    for n in tqdm(range(N)):
        for m in range(1,M):
            U[m,n+1] = r*s*U[m,n]**(s-1)*(U[m-1,n]-2*U[m,n]+U[m+1,n]) \
            + (s*(s-1)*r)/(4)*U[m,n]**(s-2)*(-U[m-1,n] + U[m+1,n])**2 + U[m,n]
            
    return x,t,U

def BTCS(f,M,N,T,xstart,xstop,s,g0,g1,init = 'f'):
    """
    Backward in time, central in space
    """
    x = np.linspace(xstart,xstop,M+1)
    t = np.linspace(0.1,T,N+1)
    U = np.zeros((M+1,N+1))

    if init == 'BB':
        _,_,Ut = refsolutionBB(s,M,N,xstart,xstop,T)
        U[:,0] = Ut[:,0]
    #IV
    else:    
        U[:,0] = f(x)
    
    #BCs
    U[0,:], U[-1,:] = g0(t),g1(t)
    
    
    h = (x[-1]-x[0])/M    # Stepsize in space
    k = T/N               # Stepsize in time

    r = k/(h**2)

    print('r = {:.2f},h = {:.2f}, k = {:.2f}'.format(r,h,k))

    for n in tqdm(range(N)):

        def func(var):
            lhs = U[2:-2,n]- var[2:-2] + s*r*var[2:-2]**(s-1)*(var[3:-1] -2*var[2:-2] + var[1:-3]) \
                +0.25*s*r*(s-1)*var[2:-2]**(s-2)*(var[3:-1]**2 -2*var[3:-1]*var[1:-3] + var[1:-3]**2)
                
            lhs = np.insert(lhs,0,U[1,n]- var[1] + s*r*var[1]**(s-1)*(var[2] -2*var[1]) \
                +0.25*s*r*(s-1)*var[1]**(s-2)*(var[2]**2))
            
            lhs = np.insert(lhs,-1,U[-2,n]- var[-2] + s*r*var[-2]**(s-1)*(var[-3]-2*var[-2]) \
                +0.25*s*r*(s-1)*var[-2]**(s-2)*(var[-3]**2))
            
            lhs = np.insert(lhs,0,var[0])
            lhs = np.insert(lhs,-1,var[-1])
            return lhs
        initial = U[:,n]
        U[:,n+1] = scipy.optimize.fsolve(func,initial)
    return x,t,U

    

def convergence_time(scheme,f,M,N,T,s,xstart,xstop,g0,g1,init):
    """
    Convergence plot in time, calculated with 1st norm with reference solution
    """
    error1 = []
    Nvec = [1000,2000,4000,8000]
    stepsizes = []
    _,_,Uref = refsolutionBB(s,M,N,xstart,xstop,T)
    for element in Nvec:
        stepsize = T/element
        stepsizes.append(stepsize)
        _,_,U = scheme(f,M,element,T,xstart,xstop,s,g0,g1,init)
        err = (U[:,0::int(element/N)]-Uref)
        err1norm = np.linalg.norm(err[int(M/2),:],1)
        error1.append(stepsize*err1norm)
    return error1,stepsizes

def convergence_space_ref(scheme,f,T,N,s,xstart,xstop,g0,g1,init,Mref = 384):
    error = []
    Mvec = [24,48,96,192] 
    stepsizes = []
    _,_,Uref = scheme(f,Mref,N,T,xstart,xstop,s,g0,g1,init)
    for element in Mvec:
        stepsize = (xstop-xstart)/element
        stepsizes.append(stepsize)
        _,_,U = scheme(f,element,N,T,xstart,xstop,s,g0,g1,init)
        factor = int(Mref/element)
        err = U[int(np.ceil(3*(element/8))):int(np.ceil(5*(element/8))),:] \
        - Uref[int(np.ceil(3*(Mref/8))):int(np.ceil(5*(Mref/8))):factor,:]
        ## slicing the solutions, to consider error in smooth areas
        cerr1 = np.linalg.norm(err[:,-1],1)
        error.append(stepsize*cerr1)
        
    return error,stepsizes


if __name__ == "__main__":
    print('Main')
    