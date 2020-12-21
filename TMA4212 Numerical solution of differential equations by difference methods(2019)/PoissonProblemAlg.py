import numpy as np   
import numpy.linalg as la   
import matplotlib.pyplot as plt 
from scipy import sparse
from scipy.sparse.linalg import spsolve 
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm
import time
newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 14}
plt.rcParams.update(newparams)

#### PROBLEM 1 ###

def A_laplace(M):
    # Construct the discrete laplacian, that is
    # A = blocktridiag(I,T,I) is a M^2xM^2 matrix where
    # T = tridiag(1,-4,1) is a MxM matrix
    M2 = M**2
    A = -4*np.eye(M2)            # The diagonal matrix
    for i in range(M2-1):
        A[i,i+1] = 1
        A[i+1,i] = 1
    for i in range(M-1,M2-1,M):
        A[i,i+1] = 0
        A[i+1,i] = 0
    for i in range(M2-M):      # The block sub- and sup diagonal
        A[i,i+M] = 1
        A[i+M,i] = 1
    return A

def plot_solution(x, y, U, txt='Solution', nfig = 1):
    # Plot the solution of the heat equation on the unit square
    fig = plt.figure(nfig)
    plt.clf()
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, U, cmap=cm.coolwarm)
    ax.view_init(azim=30)              # Rotate the figure
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(txt);
    plt.show()

def poisson_dirichlet(M):
    # Solve the Poisson equation by the 5-point formula on the unit square, with
    # Dirichlet boundary conditions.
    # Input: M, the number of interval

    # Set up the problem
    # Boundary conditions 
    def gs(x):                  # y=0
        return x**5
    def gn(x):                  # y=1
        return x**5 +3
    def gw(y):                  # x=0
        return 3*y**4
    def ge(y):                  # x=1
        return 1 + 3*y**4

    # Source function
    def f(x,y):
        return 20*x**3 + 36*y**2

    # The grid
    x = np.linspace(0, 1, M+1)
    y = np.linspace(0, 1, M+1) 
    h = 1/M

    # Inner grid
    xi = x[1:-1]       
    yi = y[1:-1] 
    Xi, Yi = np.meshgrid(xi, yi)
    Mi = M-1       # Number of inner gridpoints
    Mi2 = Mi**2    # Total number of inner gridpoints (and unknowns)

    A = A_laplace(M-1)

    # The right hand side
    b = np.zeros(Mi2)
    
    ### Boundary conditions ###
    
    b[0:Mi] = b[0:Mi]-gs(xi)                         # y=0
    b[Mi2-Mi:Mi2] = b[Mi2-Mi:Mi2] - gn(xi)           # y=1
    b[0:Mi2:Mi] = b[0:Mi2:Mi] - gw(yi)               # x=0
    b[Mi-1:Mi2:Mi] = b[Mi-1:Mi2:Mi] - ge(yi)         # x=1

    # Source function
    
    b = b+h**2*f(Xi,Yi).flatten()
    

    # Solve the linear system
    Ui = la.solve(A, b)      

    # Make an (M+1)x(M+1) array to store the solution,
    # including boundary
    U = np.zeros((M+1, M+1))

    # Reshape the solution vector, and insert into the solution matrix
    U[1:-1,1:-1] = np.reshape(Ui, (Mi,Mi)) 

    # include the boundaries
    U[0,:] = gs(x)
    U[M,:] = gn(x)
    U[:,0] = gw(y)
    U[:,M] = ge(y)

    return x, y, U

def sparse_demo(M):
    # Demonstate the benefits of using a sparse solver, by constructing the
    # discrete laplacian A, solve the system Ax=b for some vector b, and
    # compare the time used for solving the system by a full and a sparse
    # solver. 

    # NB! This solves a system of M^2 equations. 

    A = A_laplace(M)
    b = np.ones(M**2)   # Or some other arbitrary b-vector

    start = time.time()
    U = la.solve(A, b)            
    finish= time.time()
    print('Time used for solving a full system:    {:8.4f} sec'.format(finish-start))


    # Sparse solver
    As = sparse.csr_matrix(A)   # Convert the matrix to a sparse format (csr) 
    start = time.time()
    U = spsolve(As, b)          # Use a sparse linear solver
    finish = time.time()
    print('Time used for solving a sparse system:    {:8.4f} sec'.format(finish-start))


def error(u,U):
    return np.linalg.norm(u-U,np.inf)

def convergence_x(P):
    M = [2**i for i in range(1,P+1)]
    print(M)
    h = [1/M[i] for i in range(P)]
    errArr = np.zeros(P)
    for i in range(P):
        x,y,U = poisson_dirichlet(M[i])
        X,Y = np.meshgrid(x,y)
        u = X**5 + 3*Y**4
        
        # measure error at max error
        maxerr = 0
        for j in range(len(U[:,0])):
            cerr = error(u[:,j],U[:,j])
            if cerr > maxerr:
                maxerr = cerr       
        errArr[i] = maxerr
    return h,errArr 

#### PROBLEM 2 ###### 

def A_laplace(M):
    # Construct the discrete laplacian, that is
    # A = blocktridiag(I,T,I) is a M^2xM^2 matrix where
    # T = tridiag(1,-4,1) is a MxM matrix
    M2 = M**2
    A = -4*np.eye(M2)  # The diagonal matrix
    for i in range(M2-1):
        A[i,i+1] = 1
        A[i+1,i] = 1
    for i in range(M-1,M2-1,M):
        A[i,i+1] = 0
        A[i+1,i] = 0
    for i in range(M2-M):      # The block sub- and sup diagonal
        A[i,i+M] = 1
        if i < M2-2*M:
            A[i+M,i] = 1
        else:
            A[i+M,i] = 2
    #A[0:3] = 0
    #A[:,0] = 0
    #A[:,-1] = 0
    return A

def plot_solution(x, y, U, txt='Solution', nfig = 1):
    # Plot the solution of the heat equation on the unit square
    fig = plt.figure(nfig)
    plt.clf()
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, U, cmap=cm.coolwarm)
    ax.view_init(azim=30)              # Rotate the figure
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(txt);
    plt.show()

def poisson_dirichlet(M):
    # Solve the Poisson equation by the 5-point formula on the unit square, with
    # Dirichlet boundary conditions.
    # Input: M, the number of interval

    # Set up the problem
    # Boundary conditions 
    def gs(x):                  # y=0
        return x**5
    def gn(x):                  # y=1
        return x**5 + 3
    def gw(y):                  # x=0
        return 3*y**4
    def ge(y):                  # x=1
        return 1+3*y**4   

    # Source function
    def f(x,y):
        return 20*x**3 + 36*y**2

    # The grid
    x = np.linspace(0, 1, M+1)
    y = np.linspace(0, 1, M+1) 
    h = 1/M

    # Inner grid
    xi = x[1:-1]       
    yi = y[1:-1] 
    Xi, Yi = np.meshgrid(xi, yi)
    Mi = M-1       # Number of inner gridpoints
    Mi2 = Mi**2    # Total number of inner gridpoints (and unknowns)

    A = A_laplace(M-1)
    As = sparse.csr_matrix(A)
    
    # The right hand side
    b = np.zeros(Mi2)
    # Boundary conditions
    b[0:Mi] = b[0:Mi]-gs(xi)                         # y=0
    b[Mi2-Mi:Mi2] = b[Mi2-Mi:Mi2] + 2*h*12  # y=1
    b[0:Mi2:Mi] = b[0:Mi2:Mi] - gw(yi)               # x=0
    b[Mi-1:Mi2:Mi] = b[Mi-1:Mi2:Mi] - ge(yi)         # x=1

    # Source function
    b = b+h**2*f(Xi,Yi).flatten()

    # Solve the linear system
    Ui = spsolve(As, b) 

    # Make an (M+1)x(M+1) array to store the solution,
    # including boundary
    U = np.zeros((M+1, M+1))

    # Reshape the solution vector, and insert into the solution matrix
    U[1:-1,1:-1] = np.reshape(Ui, (Mi,Mi)) 

    # include the boundaries
    U[0,:] = gs(x)
    U[M,:] = gn(x)
    U[:,0] = gw(y)
    U[:,M] = ge(y)

    return x, y, U

def convergence():
    error = []
    h = [(1/2**i) for i in range (1,6)]
    for element in h:
        maxerr = 0
        dim = int(1/element)
        x,y,U = poisson_dirichlet(dim)
        xi,yi = np.meshgrid(x, y)
        Ue = xi**5 + 3*yi**4
        err = U-Ue
        for i in range(len(err[:,0])):
            print(i)
            cerr = np.linalg.norm(err[i,:],np.inf)
            if cerr > maxerr:
                maxerr = cerr
        error.append(maxerr)
    print(error)
    print(h)
    plt.loglog(h,error)
    plt.show()    
        
    
    
def sparse_demo(M):
    # Demonstate the benefits of using a sparse solver, by constructing the
    # discrete laplacian A, solve the system Ax=b for some vector b, and
    # compare the time used for solving the system by a full and a sparse
    # solver. 

    # NB! This solves a system of M^2 equations. 

    A = A_laplace(M)
    b = np.ones(M**2)   # Or some other arbitrary b-vector

    start = time.time()
    U = la.solve(A, b)            
    finish= time.time()
    print('Time used for solving a full system:    {:8.4f} sec'.format(finish-start))


    # Sparse solver
    As = sparse.csr_matrix(A)   # Convert the matrix to a sparse format (csr) 
    start = time.time()
    U = spsolve(As, b)          # Use a sparse linear solver
    finish = time.time()
    print('Time used for solving a sparse system:    {:8.4f} sec'.format(finish-start))

### PROBLEM 3 ##### 
def A_laplace(M):
    # Construct the discrete laplacian, that is
    # A = blocktridiag(I,T,I) is a M^2xM^2 matrix where
    # T = tridiag(1,-4,1) is a MxM matrix
    M2 = M**2
    A = -4*np.eye(M2)            # The diagonal matrix
    for i in range(M2-1):
        A[i,i+1] = 1
        A[i+1,i] = 1
    for i in range(M-1,M2-1,M):
        A[i,i+1] = 0
        A[i+1,i] = 0
    for i in range(M2-M):      # The block sub- and sup diagonal
        A[i,i+M] = 1
        A[i+M,i] = 1
    return A

def plot_solution(x, y, U, txt='Solution', nfig = 1):
    # Plot the solution of the heat equation on the unit square
    fig = plt.figure(nfig)
    plt.clf()
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, U, cmap=cm.coolwarm)
    ax.view_init(azim=30)              # Rotate the figure
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(txt);
    plt.show()

def poisson_dirichlet(M):
    # Solve the Poisson equation by the 5-point formula on the unit square, with
    # Dirichlet boundary conditions.


    # Set up the problem
    # Boundary conditions 
    def g3(x):
        return 1
    def g1(x):
        return 0


    # The grid
    h = 1/M
    intervals =int(6/h)
    points = int(intervals + 1)
    x = np.linspace(-3, 3, points)
    y = np.linspace(-3, 3, points) 
    

    # Inner grid
    xi = x[1:-1]       
    yi = y[1:-1] 
    Xi, Yi = np.meshgrid(xi, yi)
    Mi = points - 2    # Number of inner gridpoints
    Mi2 = Mi**2    # Total number of inner gridpoints (and unknowns)

    b = np.zeros(Mi2)
    # Boundary conditions
    b[0:Mi] = b[0:Mi]-g3(xi)                         # y=-3
    b[Mi2-Mi:Mi2] = b[Mi2-Mi:Mi2] - g3(xi)           # y=3
    b[0:Mi2:Mi] = b[0:Mi2:Mi] - g3(yi)               # x=-3
    b[Mi-1:Mi2:Mi] = b[Mi-1:Mi2:Mi] - g3(yi)         # x=3
    
    A = A_laplace(Mi)
    steps = int(2/h)
    innergridpoints = [] #index of inner gridpoints
    for i in range((steps-1),2*steps):
        for j in range(steps+1):
            innergridpoints.append(i*points + steps + j - (1+i*2))
    print(innergridpoints)
    A = np.delete(A,innergridpoints,0)
    A = np.delete(A,innergridpoints,1)
    b = np.delete(b,innergridpoints)

    As = sparse.csr_matrix(A)
    
    # The right hand side

    


    # Solve the linear system
    Ui = spsolve(As, b)

    
    UiRB = np.zeros(Mi**2) #rebuild
    count = 0
    for i in range(len(UiRB)):
        if i in innergridpoints:
            pass
        else: 
            UiRB[i] = Ui[count]
            count += 1

    # Make an (M+1)x(M+1) array to store the solution,
    # including boundary
    U = np.zeros((points, points))
    
    # Reshape the solution vector, and insert into the solution matrix
    U[1:-1,1:-1] = np.reshape(UiRB, (Mi,Mi))

    U[0,:] = g3(x)
    U[intervals,:] = g3(x)
    U[:,0] = g3(y)
    U[:,intervals] = g3(y)

    
    return x, y, U, A, b
   
        
def convergence():
    error = []
    _, _, refsol,_,_ = poisson_dirichlet(24)
    h = [6,8,12]
    stepsizes = []
    for element in h:
        stepsizes.append(1/element)
        maxerr = 0
        x,y,U,_,_ = poisson_dirichlet(element)
        err = U-refsol[0::int(24/element),0::int(24/element)]
        for i in range(len(err[:,0])):
            cerr = np.linalg.norm(err[i,:],np.inf)
            if cerr > maxerr:
                maxerr = cerr
        error.append(maxerr)
    print(error)
    print(stepsizes)
    plt.title('Convergence plot')
    rate1 = np.polyfit(np.log(stepsizes),np.log(error),1)
    print(rate1)
    plt.loglog(stepsizes,error,'r-',label='Convergence rate: '+str(rate1[0]))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Stepsize [1/N]')
    plt.ylabel('Error')
    plt.xticks(stepsizes[:-1])
    plt.yticks(error[:-1])
    plt.show()  
    
def sparse_demo(M):
    # Demonstate the benefits of using a sparse solver, by constructing the
    # discrete laplacian A, solve the system Ax=b for some vector b, and
    # compare the time used for solving the system by a full and a sparse
    # solver. 

    # NB! This solves a system of M^2 equations. 

    A = A_laplace(M)
    b = np.ones(M**2)   # Or some other arbitrary b-vector

    start = time.time()
    U = la.solve(A, b)            
    finish= time.time()
    print('Time used for solving a full system:    {:8.4f} sec'.format(finish-start))


    # Sparse solver
    As = sparse.csr_matrix(A)   # Convert the matrix to a sparse format (csr) 
    start = time.time()
    U = spsolve(As, b)          # Use a sparse linear solver
    finish = time.time()
    print('Time used for solving a sparse system:    {:8.4f} sec'.format(finish-start))


'''
if __name__ == "__main__":
    x, y, U = poisson_dirichlet(10)
    X, Y = np.meshgrid(x,y)
    
    U_exact = X**5 + 3*Y**4
    plot_solution(x, y, U)
    plot_solution(x, y, U-U_exact, txt='Error', nfig=2)
    
    h,errx = convergence_x(7)
    ratex = np.polyfit(np.log(h),np.log(errx),1)
    print(ratex[0])

    plt.figure()
    plt.title('Convergence plot')
    plt.xlabel(r'$\logh$')
    plt.ylabel(r'$\log|e_h|$')
    plt.loglog(h,errx,'r.--',label= 'rate = %.2f'%(ratex[0]))

    plt.legend()
'''    
