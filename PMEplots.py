#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:12:53 2019

@author: emilam
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  


###################################################################
newparams = {'figure.figsize': (7.0, 5.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 12}
plt.rcParams.update(newparams)
###################################################################

def tridiag(a, b, c, N):
    # Returns a tridiagonal matrix A=tridiag(c, a, b) of dimension N x N.
    e = np.ones(N)        # array [1,1,...,1] of length N
    A = a*np.diag(e[1:],-1)+b*np.diag(e)+c*np.diag(e[1:],1)
    return A


def plot_solution(x, t, U, txt='Solution'):
    # Plot the solution of the heat equation
    fig = plt.figure()
    plt.clf()
    ax = fig.gca(projection='3d')
    T, X = np.meshgrid(t,x)
    # ax.plot_wireframe(T, X, U)
    ax.plot_surface(T, X, U, cmap=cm.jet)
    ax.view_init(azim=30)     # Rotate the figure
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u(x,t)')
    plt.title(txt);

def plot_solution_angles(x, t, U, txt='Solution'):
    # Plot the solution of the heat equation
    #fig,(ax1,ax2,ax3,ax4) = plt.subplots(2, 2, sharex='all')
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2,2,2,projection = '3d')
    ax3 = fig.add_subplot(2,2,3,projection = '3d')
    ax4 = fig.add_subplot(2,2,4,projection = '3d')
    T, X = np.meshgrid(t,x)
    # ax.plot_wireframe(T, X, U)
    
    ax1.plot_surface(T, X, U, cmap=cm.coolwarm)
    ax1.view_init(azim=240, elev = 15)
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('u')
    
    ax2.plot_surface(T,X,U,cmap = cm.coolwarm)
    ax2.view_init(azim=120, elev = 30)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u')
    
    
    ax3.plot_surface(T,X,U,cmap = cm.coolwarm)
    ax3.view_init(azim = 180,elev = 0)
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_zlabel('u')
    
    
    ax4.plot_surface(T,X,U,cmap = cm.coolwarm)
    ax4.view_init(azim = 270,elev = 90)
    ax4.set_xlabel('t')
    ax4.set_ylabel('x')
    ax4.set_zlabel('u')
    
    fig.suptitle('Solution to the model problem')
    
if __name__ == "__main__":
    print('Main')
    