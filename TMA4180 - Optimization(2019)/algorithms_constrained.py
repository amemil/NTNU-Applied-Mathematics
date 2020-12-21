#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:44:29 2019

@author: ronald
"""

def gradient_descent_augL(f, grad, updateL, x0, tol, mu_k, l_k):
    """
    Gradient descent algorithm
    """
    cc1,cc2,alpha0 = 0.25, 0.5, 1
    p_k = -grad(x0, l_k, mu_k)
    x_k = x0
    x_k_ = x0 + np.ones((6,1))*100
    Nmax = 60
    
    points_x = [x0[tp0]]
    points_y = [x0[tp1]]
    
    it = 0
    while la.norm(p_k) > tol and it < Nmax:
        #print(x_k)
        p_k /= la.norm(p_k)
        alpha = backtracking_linesearch_augL(f,l_k, mu_k, grad,p_k,x_k)
        #print(alpha)
        #alpha = bisection_linesearch_augL(f, grad, x_k, p_k, l_k, mu_k, 100, 0.1, 0.7)
        l_k = updateL(l_k, x_k, mu_k)
        mu_k = updateMu(mu_k)
        x_k = x_k+alpha*p_k
        p_k = -grad(x_k, l_k, mu_k)
        
        points_x = points_x + [x_k[tp0]]
        points_y = points_y + [x_k[tp1]]
        it += 1
        #print(p_k)
        if plot_all:
            xd = x_k + p_k/la.norm(p_k,2)
            fplot = lambda x, y: f3augL(np.array([x, y, x_k[2], x_k[3], x_k[4], x_k[5]]), w, z, l_k, mu_k)
            plt.figure(it+10)
            
            plot_level_curves(fplot,[-4, 4],[-4,4], 'contour', 'faug3','None', points_x, points_y, 0.5)
    
            plt.plot([x_k[tp0], xd[tp0]], [x_k[tp1], xd[tp1]], color='r', linewidth=1)
            plt.show()
            plt.figure(1)
            cc1, rr1, cc2, rr2 = x2circles(x_k)
            plot_result('it', data_xrange, data_yrange, w, z, cc1, rr1, cc2, rr2)
            #plt.show()
            
            plt.figure(it+12)
            plt.clf()
            ran = 0.3
            xd = x_k + p_k/la.norm(p_k,2)*ran
            x_left, x_right = x_k[tp0]-ran, x_k[tp0]+ran
            y_left, y_right = x_k[tp1]-ran, x_k[tp1]+ran
            plot_level_curves(fplot,[x_left, x_right],[y_left,y_right], '3D', 'faug3','None', [x_k[tp0]], [x_k[tp1]], 0.05)
            plt.plot([x_k[tp0], xd[tp0]], [x_k[tp1], xd[tp1]], color='r', linewidth=1)
            
            plt.show()
        if la.norm(x_k - x_k_,2)<tol:
            print('converged')
            return x_k,it,f(x_k, l_k, mu_k), l_k, mu_k, points_x, points_y
        x_k_ = x_k
        
    return x_k,it,f(x_k, l_k, mu_k), l_k, mu_k, points_x, points_y

def backtracking_linesearch_augL(f,l, mu, gradf,p,x):
    rho = 0.5
    c = 1e-4
    a = 1
    Nmax = 1000
    
    f0 = f(x, l, mu)
    phi = f(x+a*p, l, mu)
    dF = np.array(gradf(x, l, mu))
    
    it = 0
    
    while (phi >= (f0 + c*a*dF.dot(p)) and it < Nmax):
        a = rho*a
        phi = f(x+a*p, l, mu)
        it += 1
    return a

def bisection_linesearch_augL(f,df,x,p,l, mu, alpha0,bc1,bc2):
    """Bisection algorithm calculating a step length
        satisfying the Wolfe conditions
    """
    alpha = alpha0
    alpha_min = 0
    alpha_max = np.inf
    fx = f(x, l, mu)
    dfx = df(x, l, mu)
    dfxp = dfx.dot(p)
    
    Nmax = 100
    it = 0
    while alpha < 1e+10:
        if f(x + alpha*p, l, mu) > fx + bc1*alpha*dfxp and it < Nmax:
            # No sufficient decrease: Too short step
            alpha_max = alpha
            alpha = 0.5*(alpha_min + alpha_max)
            it +=1
        elif df(x + alpha*p, l, mu).dot(p) < bc2*dfxp and it < Nmax:
            # No curvature: Too short step
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha *= 2.0
                it+=1
            else:
                alpha = 0.5*(alpha_min + alpha_max)
                it+=1
        else:
            print('returning', alpha)
            return alpha
    raise ValueError('Steplength is too long!')

def updateL(l, x, mu):
    l[0] = l[0]- mu*co0(x)
    l[1] = l[1]- mu*co1(x)
    l[2] = l[2] -mu*co2(x)
    return l

def updateMu(mu):
    return mu*100