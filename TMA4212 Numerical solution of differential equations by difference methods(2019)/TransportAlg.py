# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt


def transport_ftbs(f,g,M,N,T,r):
    U = np.zeros((M+1,N+1))
    x = np.linspace(0,3,M+1)
    t = np.linspace(0,T,N+1)
    
    #IV
    U[0,:] = g(t)
    
    #BC
    U[:,0] = f(x)
    
    for n in range(N):
        for m in range(1,M+1):
            U[m,n+1] = U[m,n]-r*(U[m,n]-U[m-1,n])
    return U,x,t

def transport_lw(f,g,M,N,T,r):
    U = np.zeros((M+1,N+1))
    x = np.linspace(0,3,M+1)
    t = np.linspace(0,T,N+1)
    
    #IV
    U[0,:] = g(t)
    
    #BC
    U[:,0] = f(x)
    for n in range(N):
        for m in range(1,M):
            U[m,n+1] = U[m,n]-(r/2)*(U[m+1,n]-U[m-1,n])+((r**2)/2)*(U[m+1,n]-2*U[m,n]+U[m-1,n])
    return U,x,t

def transport_w(f,g,M,N,T,r):
    U = np.zeros((M+1,N+1))
    x = np.linspace(0,3,M+1)
    t = np.linspace(0,T,N+1)
    
    #IV
    U[0,:] = g(t)
    
    #BC
    U[:,0] = f(x)
    for n in range(N):
        for m in range(1,M+1):
            U[m,n+1] = U[m-1,n]-((1-r)/(1+r))*(U[m-1,n+1]-U[m,n])
    return U,x,t
    
def plot(x,U,t,tplot,title):
    k = t[1] - t[0]
    plt.figure()
    plt.clf()
    for tn in tplot:
        n = int(tn/k)
        tn = n*k
        plt.plot(x,U[:,n],label = 't={:.1f}'.format(tn))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Solution')
    plt.title(title)
    
    

if __name__ == "__main__":
    print('main function')
    
    def g(t):
        return np.ones(len(t))
    
    def f(x):
        return np.zeros(len(x))
    
    def exact(x,t,a):
        U = np.zeros((len(x),len(t)))
        for i in range(len(x)):
            for j in range(len(t)):
                if x[i] < a*t[j]:
                    U[i,j] = 1
                else:
                    U[i,j] = 0
        return U
    a = 1
    h = 1/160
    Tmax = 2
    
    #case 1: r = 1 --> exact solution
    k1 = h/a
    #case 2: r = 0.5
    k2 = h/(2*a)
    
    r1 = a*k1/h
    r2 = a*k2/h
    
    M = int(3/h)
    N1 = int(Tmax/k1)
    N2 = int(Tmax/k2)
    #FTBS
    U1,x1,t1 = transport_ftbs(f,g,M,N1,Tmax,r1)
    U2,x2,t2 = transport_ftbs(f,g,M,N2,Tmax,r2)
    
    #Lax-Wendroff
    U3,x3,t3 = transport_lw(f,g,M,N1,Tmax,r1)
    U4,x4,t4 = transport_lw(f,g,M,N2,Tmax,r2)
    
    #Wendroff
    U5,x5,t5 = transport_w(f,g,M,N1,Tmax,r1)
    U6,x6,t6 = transport_w(f,g,M,N2,Tmax,r2)
    
    #Exact solution
    Uex = exact(x1,t1,a)
    
    tplots = [0.0,0.5,1.0,1.5,2.0]
    
    k = t1[1] - t1[0]
     
    ### PLOTTING ###

    
    #Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)


    for tn in tplots:
        n = int(tn/k)
        tn = n*k
        ax1.plot(x1,Uex[:,n])#,label = 't={:.1f}'.format(tn))
    ax1.legend()
    ax1.set_ylim([-0.2,1.2])
    ax1.set_xlabel('x')
    ax1.set_ylabel('Solution')
    ax1.set_title('Exact solution')

    for tn in tplots:
        n = int(tn/k)
        tn = n*k
        ax2.plot(x1,U1[:,n])#,label = 't={:.1f}'.format(tn))
        ax2.plot(x3,U3[:,n])#,label = 't={:.1f}'.format(tn))
        ax2.plot(x5,U5[:,n])#,label = 't={:.1f}'.format(tn))
    ax2.set_ylim([-0.2,1.2])
    #ax2.set_ylabel('x')
    ax2.set_xlabel('x')
    ax2.set_title('Numerical solutions (r=1)')
    ax2.legend()
    
      #Creates two subplots and unpacks the output array immediately
    f2, (ax21, ax22, ax23) = plt.subplots(1, 3, sharey=False)


    for tn in tplots:
        n = int(tn/k)
        tn = n*k
        ax21.plot(x2,U2[:,n])#,label = 't={:.1f}'.format(tn))
    ax21.legend()
    ax21.set_ylim([-0.05,1.05])
    ax21.set_xlabel('x')
    ax21.set_ylabel('Solution')
    ax21.set_title('FTBS (r=0.5)')

    for tn in tplots:
        n = int(tn/k)
        tn = n*k
        ax22.plot(x4,U4[:,n])#,label = 't={:.1f}'.format(tn))
    ax22.set_ylim([-0.05,1.35])
    #ax2.set_ylabel('x')
    ax22.set_xlabel('x')
    ax22.set_title('Lax-Wendroff (r=0.5)')
    ax22.legend()
    
    for tn in tplots:
        n = int(tn/k)
        tn = n*k
        ax23.plot(x6,U6[:,n],label = 't={:.1f}'.format(tn))
    ax23.set_ylim([-0.3,1.05])
    #ax2.set_ylabel('x')
    ax23.set_xlabel('x')
    ax23.set_title('Wendroff (r=0.5)')
    ax23.legend(fontsize=8.5)
    
    plot(x1,U1,t1,tplots,'FTBS: Numerical soltion r ={:.1f}'.format(r1))
    plot(x2,U2,t2,tplots,'FTBS: Numerical solution r = {:.1f}'.format(r2))
    plot(x3,U3,t3,tplots,'Lax-Wendroff: Numerical soltion r ={:.1f}'.format(r1))
    plot(x4,U4,t4,tplots,'Lax-Wendroff: Numerical solution r = {:.1f}'.format(r2))
    plot(x5,U5,t5,tplots,'Wendroff: Numerical soltion r ={:.1f}'.format(r1))
    plot(x6,U6,t6,tplots,'Wendroff: Numerical solution r = {:.1f}'.format(r2))
    plot(x1,Uex,t1,tplots,'Exact solution')
    