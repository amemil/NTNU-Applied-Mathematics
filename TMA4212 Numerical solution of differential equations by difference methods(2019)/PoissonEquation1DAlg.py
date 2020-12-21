'''
TMA4212 Numerical solution of partial differential equations by difference methods. 
Exercise 1.
'''
import numpy as np
import numpy.linalg as la
from numpy import inf
import matplotlib.pyplot as plt

# The following is some settings for the figures. 
# This can be manipulated to get nice plots included in pdf-documents.
newparams = {'figure.figsize': (8.0, 5.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 12}
plt.rcParams.update(newparams)


def tridiag(c, a, b, N):
    # Returns a tridiagonal matrix A=tridiag(c, a, b) of dimension N x N.
    e = np.ones(N)        # array [1,1,...,1] of length N
    A = c*np.diag(e[1:],-1)+a*np.diag(e)+b*np.diag(e[1:],1)
    return A

def bvp(f, alpha, beta, M=10):
    # Solve the BVP -u''(x)=f(x), u(0)=alpha, u(1)=beta
    # by a central difference scheme. 
    h = 1/M
    Ah = tridiag(-1,2,-1,M-1)/h**2      # Set up the coefficient matrix

    x = np.linspace(0,1,M+1)    # gridpoints, including the boundary points
    xi = x[1:-1]             # inner gridpoints 
    F = f(xi)                # evaluate f in the inner gridpoints
    F[0] = F[0]+alpha/h**2   # include the contribution from the boundaries
    F[-1] = F[-1]+beta/h**2

    # Solve the linear equation system
    Ui = la.solve(Ah, F)        # the solution in the inner gridpoints   

    # Include the boundary points in the solution vector 
    Ue = exactf(x)
    U = np.zeros(M+1)
    U[0] = alpha
    U[1:-1] = Ui
    U[-1] = beta
    return x, U, Ue

def bvp2(f, alpha, M=10):
    # Solve the BVP -u''(x)=f(x), u(0)=alpha, u(1)=beta
    # by a central difference scheme. 
    h = 1/M
    Ah = tridiag(-1,2,-1,M-1)/h**2      # Set up the coefficient matrix
    x = np.linspace(0,1,M+1)    # gridpoints, including the boundary points
    xi = x[1:-1]                # inner gridpoints 
    #beta = 1-beta1(x[-1],x[-2],h)      
    #beta = 1-beta2(x[-1],x[-2],x[-3],h)
    beta = 1-beta3(x[-1]+h,x[-2],h)
    F = f(xi)                # evaluate f in the inner gridpoints
    F[0] = F[0]+alpha/h**2   # include the contribution from the boundaries
    F[-1] = F[-1]+beta/h**2

    # Solve the linear equation system
    Ui = la.solve(Ah, F)        # the solution in the inner gridpoints   

    # Include the boundary points in the solution vector 
    Ue = exactf(x)
    U = np.zeros(M+1)
    U[0] = alpha
    U[1:-1] = Ui
    U[-1] = beta
    return x, U, Ue

# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Set up the problem to be solved. 
    def f(x):                  # the right hand side
        y = 6*x
        return y
    alpha, beta = 0, 1         # boundary values
        
    def exactf(x):
        return -x**3 + 2*x
    
    def f2(x):                  # the right hand side
        return np.sin(np.pi*x)
    
    alpha = 0         # boundary values
    
    def beta1(M1,M0,h):
        return (exactf(M1)-exactf(M0))/h
    
    def beta2(M2,M1,M0,h):
        return (3*exactf(M2)-4*exactf(M1)+exactf(M0))/(2*h)
        
    def beta3(M2,M0,h):
        return (exactf(M2)-exactf(M0))/(2*h)
    
    def exactf2(x):
        return np.sin(np.pi*x)/np.pi**2 + ((np.pi+1)/(2*np.pi))*x
    
    # Solve the BVP
    x, U, Ue = bvp(f, alpha, beta, M=10)


    
    # And plot the solution
    plt.plot(x,U,'-x',label='Exact')
    plt.plot(x,Ue,label='Approx')
    plt.plot(x,U-Ue,label='Error')
    plt.xlabel('x')
    plt.ylabel('U')
    plt.title('Numerical solution of the model problem')
    plt.legend()
    plt.show()
