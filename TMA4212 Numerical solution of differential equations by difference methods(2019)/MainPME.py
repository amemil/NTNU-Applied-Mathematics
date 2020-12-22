#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:13:59 2019

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
import PorousMediumAlg as alg
import PMEplots as plot

if __name__ == "__main__":
    print('Main')
    
    def f(x):
        """
        Initial value, dirac delta source for barenblatts solution
        """
        f = np.zeros(len(x))
        f = np.exp(-x**2)
        return f 
          
    def g0(t):
        "Left BC"
        return 0
    
    def g1(t):
        "Right BC"
        return 0
    
    M = 100
    N = 800
    T = 2
    s=2
    init = 'BB'
    method = 'Exact'
    eq = 1 
    xstart= -4
    xstop = 4
    
    ### PLOTTING EXACT AND NUMERICAL SOLUTIONS ###
    xb,tb,Ub = alg.refsolutionBB(s,100,50,xstart,xstop,T,lam = 0.4, n=1)
    xe,te,Ue = alg.FTCS(f,100,800,T,xstart,xstop,s,g0,g1,init = 'BB')
    xi,ti,Ui = alg.BTCS(f,200,200,T,xstart,xstop,s,g0,g1,init = 'BB')
    
    plot.plot_solution(xb,tb,Ub,'Barenblatt solution')
    plot.plot_solution(xe,te,Ue,'FTCS-scheme')
    plot.plot_solution(xi,ti,Ui,'BTCS-scheme')
    
    ### PLOTTING CONVERGENCE PLOTS ###
    # TIME :
    error_time1,stepsizes_time1 = alg.convergence_time(alg.FTCS,f,100,50,T,s,xstart,xstop,g0,g1,init = 'BB')
    error_time2,stepsizes_time2  = alg.convergence_time(alg.BTCS,f,100,50,T,s,xstart,xstop,g0,g1,init = 'BB')
    plt.figure()
    plt.title('Convergence in time')
    rate1 = np.polyfit(np.log(stepsizes_time1),np.log(error_time1),1)
    plt.loglog(stepsizes_time1,error_time1,'--g.',label='FTCS: {:0.2f}'.format(rate1[0]))
    rate2 = np.polyfit(np.log(stepsizes_time2),np.log(error_time2),1)
    plt.loglog(stepsizes_time2,error_time2,'--r.',label='BTCS: {:0.2f}'.format(rate2[0]))
    plt.xlabel('Stepsize [k]')
    plt.ylabel('Error')
    plt.legend()
    plt.show()      
 
    #SPACE:
    error1_space,stepsizes_space1 = alg.convergence_space_ref(alg.FTCS,f,T,24000,s,xstart,xstop,g0,g1,init = 'f',Mref = 384)
    error2_space,stepsizes_space2 = alg.convergence_space_ref(alg.BTCS,f,T,800,s,xstart,xstop,g0,g1,init = 'f',Mref = 384)
    
    plt.figure()
    plt.title('Convergence plot - space')
    rate1 = np.polyfit(np.log(stepsizes_space1),np.log(error1_space),1)
    plt.loglog(stepsizes_space1,error1_space,'--g.',label='FTCS: {:0.2f}'.format(rate1[0]))
    rate2 = np.polyfit(np.log(stepsizes_space2),np.log(error2_space),1)
    plt.loglog(stepsizes_space2,error2_space,'--r.',label='BTCS: {:0.2f}'.format(rate2[0]))
    plt.xlabel('Stepsize [h]')
    plt.ylabel('Error (1st norm)')
    plt.legend()
    plt.show() 
    
    
    ### Animation of implicit method  ###
    # for explicit scheme, change xi,ti and Ui with xe,te and Ue in function animate below.

    fig = plt.figure(1)
    ax = plt.axes(xlim=(-4, 4), ylim=(0, 1.5))

    ax.set_xlabel('x')
    ax.set_ylabel('U(x,t)')
    ax.set_title('Evolution of U(x,t) with time')
    line, = ax.plot([], [], lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return time_text,line,

    # animation function.  This is called sequentially
    def animate(i):
        x = xi
        y = Ui[:,i]
        line.set_data(x,y)
        time_text.set_text('time = {:.2f}'.format(ti[i]))
        return time_text,line,
    

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=1000, interval=1, blit=True)

    
