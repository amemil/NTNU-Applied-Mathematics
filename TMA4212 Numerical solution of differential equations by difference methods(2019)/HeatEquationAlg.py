import numpy as np              
import numpy.linalg as la   
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm 
from scipy import integrate
from scipy import sparse
from scipy.sparse.linalg import spsolve 

#from scipy.integrate import solve_ivp
newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 12}
plt.rcParams.update(newparams)


### FUNCTIONS FOR EXERCISE 2 ### 
def tridiag(a, b, c, N):
    # Returns a tridiagonal matrix A=tridiag(a, b, c) of dimension N x N.
    e = np.ones(N)        # array [1,1,...,1] of length N
    A = a*np.diag(e[1:],-1)+b*np.diag(e)+c*np.diag(e[1:],1)
    return A
    

def backward_euler_matrix(f, M=10, N=100, T=0.5):
    h = 1/M     # Stepsize in space
    k = T/N     # Stepsize in time

    r = k/h**2
    # Print the stepsizes, and r=k/h^2.
    print('h={:.4f}, k={:.4f}, r={:.4f}'.format(h,k,r))

    U = np.zeros((M+1,N+1))    # Array to store the solution, boundaries included.
    x = np.linspace(0,1,M+1)   # Gridpoints on the x-axis
    t = np.linspace(0,T,N+1)   # Gridpoints on the t-axis
    U[:,0] = f(x)              # Initial values
    
    #since u(0,t)=u(1,t)=0 --> q^n = [0,...,0]
    C = np.identity(M-1) + r*tridiag(-1,+2,-1,M-1)
    Cs = sparse.csr_matrix(C)
    # Main loop 
    for n in range(N):
        U[1:-1, n+1] = spsolve(Cs,U[1:-1,n])
    return x, t, U

def crank_nicol_matrix(f, M=10, N=100, T=0.5):
    h = 1/M     # Stepsize in space
    k = T/N     # Stepsize in time

    r = k/h**2
    # Print the stepsizes, and r=k/h^2.
    print('h={:.4f}, k={:.4f}, r={:.4f}'.format(h,k,r))

    U = np.zeros((M+1,N+1))    # Array to store the solution, boundaries included.
    x = np.linspace(0,1,M+1)   # Gridpoints on the x-axis
    t = np.linspace(0,T,N+1)   # Gridpoints on the t-axis
    U[:,0] = f(x)              # Initial values
    
    #since u(0,t)=u(1,t)=0 --> q^n = [0,...,0]
    C = np.identity(M-1) + r*tridiag(-0.5,1,-0.5,M-1)
    d = np.zeros((M+1,N+1))
    Cs = sparse.csr_matrix(C)
    # Main loop 
    for n in range(N):
        for m in range(M):
            d[m,n+1] = (r/2)*U[m-1,n]+(1-r)*U[m,n]+(r/2)*U[m+1,n]
        U[1:-1, n+1] = spsolve(Cs,d[1:-1,n+1])
    return x, t, U


def crank_nicol_matrix_period(f, M=10, N=100, T=0.5):
    h = 1/M     # Stepsize in space
    k = T/N     # Stepsize in time

    r = k/h**2
    # Print the stepsizes, and r=k/h^2.
    print('h={:.4f}, k={:.4f}, r={:.4f}'.format(h,k,r))

    U = np.zeros((M+1,N+1))    # Array to store the solution, boundaries included.
    x = np.linspace(0,1,M+1)   # Gridpoints on the x-axis
    t = np.linspace(0,T,N+1)   # Gridpoints on the t-axis
    U[:-1,0] = f(x[:-1])              # Initial values
    
    
    A = tridiag(1,-2,1,M)
    A[0][-1] = 1
    A[-1][0] = 1
    
    lhs = np.identity(M) - 0.5*r*A
    rhs = np.identity(M) + 0.5*r*A
    
    lhsSmatrix = sparse.csr_matrix(lhs)
    
    # Main loop 
    for n in range(N):
        rhsU = np.matmul(rhs,U[:-1,n])
        rhsSmatrix = sparse.csr_matrix(rhsU)
        U[:-1, n+1] = spsolve(lhsSmatrix,rhsU)
        U[-1,n+1] = U[0,n+1]

    return x, t, U,T
    
def convergence_plot_h(T, N, fscheme, fexact,f1):
    hlist = []
    err  = []
    k = T/N
    for i in range(8,0,-1):
        M = (2**i)
        h = 1/M
        hlist.append(h)
        x1, t1, U1 = fscheme(f1, M=M, N=N, T=T)
        uexact = uex_1(x1,T)
        err.append(np.sqrt(h)*np.linalg.norm(uexact - U1[:,-1]))
    rate1 = np.polyfit(np.log(hlist),np.log(err),1)
    return hlist,err,rate1,N
    

def convergence_plot_k(T, M, fscheme,f1):
    klist = []
    err = []
    h = 1/M
    #_, _, refsol = fscheme(f1,M=M,N=10000,T=T)
    for i in range(1,10):
        N = (2**i)
        k = T/N
        klist.append(k)
        x1, t1, U1 = fscheme(f1,M=M,N=N,T=T)
        uexact = uex_1(x1,T)
        err.append(np.sqrt(h)*np.linalg.norm(uexact - U1[:,-1]))
    rate1 = np.polyfit(np.log(klist),np.log(err),1)
    return klist,err,rate1,M

def f1(x):
    return np.sin(np.pi*x)
        
# Exact solution for example 1:
def uex_1(x,t):  
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)
    


### NUMERICAL SOLUTIONS OF 2A) ###

x1, t1, U1 = backward_euler_matrix(f1, M=50, N=2000, T=0.5)
x2, t2, U2 = crank_nicol_matrix(f1, M=50, N=2000, T=0.5)

plt.figure(1)
plt.clf()
tplots = [0.0,0.1,0.2,0.3,0.4,t1[-1]]
k = t1[1]-t1[0]
for tn in tplots:
    n = int(tn/k)
    tn = n*k
    plt.plot(x1,U1[:,n],'o',label='t1={:.1f}'.format(tn))
    #plt.plot(x1,uex_1(x1,tn),label=)
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.title('Backward Euler')
plt.savefig('2aBE')
plt.show()

plt.figure(2)
plt.clf()
tplots = [0.0,0.1,0.2,0.3,0.4,t1[-1]]
k = t2[1]-t2[0]
for tn in tplots:
    n = int(tn/k)
    tn = n*k
    plt.plot(x2,U2[:,n],'o',label='t2={:.1f}'.format(tn))
    #plt.plot(x2,uex_1(x2,tn),label='exact')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.title('Crank-Nicolson')
plt.savefig('2aCN')
plt.show()

### CONVERGENCE PLOTS OF 2A) ###

#IN SPACE
h1,err1,rate1, N = convergence_plot_h(0.1,10000,backward_euler_matrix,uex_1,f1)
h2,err2,rate2, N = convergence_plot_h(0.1,10000,crank_nicol_matrix,uex_1,f1)

plt.figure()
plt.title('Convergence plot (N='+str(N)+')')
plt.loglog(h1,err1,'b-o',label='Backward Euler') 
plt.loglog(h2,err2,'r-o',label='Crank-Nicolson')
plt.xlabel('Stepsize [h]')
plt.ylabel('Error (2nd norm)')
plt.legend()
plt.savefig('2aspace')
plt.show()

#IN TIME
k1, errk1,ratek1,M = convergence_plot_k(0.1,2000,backward_euler_matrix,f1)
k2,errk2,ratek2,M = convergence_plot_k(0.1,2000,crank_nicol_matrix,f1)

plt.figure()
plt.title('Convergence plot (M='+str(M)+')')
plt.loglog(k1,errk1,'b-o',label='Backward Euler') 
plt.loglog(k2,errk2,'r-o',label='Crank-Nicolson')
plt.xlabel('Stepsize [k]')
plt.ylabel('Error (2nd norm)')
plt.legend()
plt.savefig('2atime')
plt.show()

### NUMERICAL SOLUTIONS OF 2B, PERIODIC BOUNDARY CONDITIONS ###


xp, tp, Up,T = crank_nicol_matrix_period(f1, M=20, N=400, T=0.1)


def uex_per(x,t): #exact solution
    u = 2/np.pi
    for n in range(1,1000):
        u += 4/(np.pi-4*np.pi*n**2)*np.cos(2*n*np.pi*x)*np.exp(-2**2*n**2*np.pi**2*t)
    return u
    


plt.figure()
plt.title('Crank-Nicolson periodic')
plt.clf()
tplots = np.linspace(0,T,5)
k = tp[1]-tp[0]
for tn in tplots:
    n = int(tn/k)
    tn = n*k
    plt.plot(xp,Up[:,n],'x-',label='tp={:.2f}'.format(tn))
    #plt.plot(x2,uex_1(x2,tn),label='exact')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.savefig('2bperiodic')
plt.show()

###CONVERGENCE PLOTS FOR 2b) ###


def convergence_h_2b(f,N,T):
    hlist = []
    err  = []
    k = T/N
    imax = 8
    #_,_,refSol,_ = crank_nicol_matrix_period(f,4*2**(imax),10000,T)
    for i in range(imax,0,-1):
        M = (2**i)
        h = 1/M
        hlist.append(h)
        x1, t1, U1,_ = crank_nicol_matrix_period(f1, M, N, T)
        exact = uex_per(x1,T)
        err.append(np.sqrt(h)*np.linalg.norm(U1[:,-1]-exact))
        #if len(refSol[:,-1]) > len(U1[:,-1]):
        #    err.append(np.linalg.norm(refSol[:len(U1[:,-1]),-1] - U1[:,-1]))
        #else:
        #    err.append(np.linalg.norm(refSol[:,-1] - U1[:len(refSol[:,-1]),-1]))
    rate1 = np.polyfit(np.log(hlist),np.log(err),1)
    return hlist,err,rate1

def convergence_k_2b(f,M,T):
    klist = []
    err  = []
    h = 1/M
    #_,_,refSol,_ = crank_nicol_matrix_period(f,M,100000,T)
    for i in range(10,0,-1):
        N = (2**i)
        k = T/N
        klist.append(k)
        x1, t1, U1,_ = crank_nicol_matrix_period(f1, M, N, T)
        exact = uex_per(x1,T)
        #err.append(np.sqrt(h)*np.linalg.norm(U1[:,-1]-refSol[:,-1]))
        err.append(np.sqrt(h)*np.linalg.norm(U1[:,-1]-exact))
    rate1 = np.polyfit(np.log(klist),np.log(err),1)
    return klist,err,rate1

#IN SPACE
hlist2b,err12b,rate12b = convergence_h_2b(f1,10000,0.1)

plt.figure()
plt.title('Convergence plot in space')
plt.loglog(hlist2b,err12b,'b-o',label='Crank-Nicolson') 
plt.xlabel('Stepsize [h]')
plt.ylabel('Error (2nd norm)')
plt.legend()
plt.savefig('2bspace')
plt.show()

#IN TIME

klist2b,err22b,rate22b = convergence_k_2b(f1,3000,0.1)

plt.figure()
plt.title('Convergence plot in time')
plt.loglog(klist2b,err22b,'b-o',label='Crank-Nicolson') 
plt.xlabel('Stepsize [k]')
plt.ylabel('Error (2nd norm)')
plt.legend()
plt.savefig('2btime')
plt.show()
plt.figure()
#plt.title('Convergence 2b')


### PROBLEM 3 ###
'''


newparams = {'figure.figsize': (7.0, 5.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 12}
plt.rcParams.update(newparams)



def plot_solution(x, t, U, txt='Solution'):
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
    ax1.view_init(azim=60, elev = 15)
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
    ax4.view_init(azim = 90,elev = 90)
    ax4.set_xlabel('t')
    ax4.set_ylabel('x')
    ax4.set_zlabel('u')
    
    fig.suptitle('Solution to the model problem')


def F(t,y):
    M = len(y)
    h = 1/M
    d = 0.10
    Q = tridiag(1,-2,1,M)
    Q[0,1] = 2
    Q[-1,-2] = 2
    
    return d/(h**2)*np.matmul(Q,y) + y*(np.ones(M)-y)
    

def solver(f,M,T):
    x = np.linspace(0,1,M)
    values = solve_ivp(F,(0,T),f(x))
        
    return values,x

if __name__ == "__main__":
    
    def f(x):
        return np.sin(np.pi*(x-0.25))**(100)
    
    values,x = solver(f,M=100,T=10)
    
    plot_solution(x,values.t,values.y,txt = 'Solution to the model problem')
    




'''








