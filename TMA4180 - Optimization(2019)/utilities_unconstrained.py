import numpy as np
import numpy.linalg as la



def make_Ac(x,d =2):
    A = np.zeros((d,d))
    c = np.zeros(d)
    
    A[0][0] = x[0]
    A[0][1],A[1][0] = x[1],x[1]
    A[1][1] = x[2]
    
    c[0] = x[3]
    c[1] = x[4]
    return A, c

def from_Ac_to_x(A,c):
    x = np.zeros(A.shape[0] + c.size + 1)
    x[0] = A[0][0]
    x[1] = A[0][1]
    x[2] = A[1][1]
    x[3] = c[0]
    x[4] = c[1]
    return x
    
def not_posdef(A):
    """
    checks whether a 2x2 matrix A is not positive definite
    """
    return ((A[0][0] <= 0 and A[1][1] <= 0 ) or la.det(A) <= 0)

def generate_random(N,d,scale):
    x = np.random.randn(5)
    z = np.random.uniform(low = -scale,high=scale,size=(N,d)) # randn
    return x,z 

def generate_random_x(N,d,scale):
    return np.random.uniform(low = -scale,high = scale,size = 5)

def generate_random_z(N,d,scale):
    return np.random.uniform(low = -scale,high = scale,size = (N,d))


def hi(x, zi,model):
    A, c = make_Ac(x,2)
    if model == 1:
        return (zi-c).dot(A.dot(zi-c)) - 1
    elif model == 2:
        b = c
        return zi.T@A@zi - zi.T@b - 1
    else:
        raise ValueError('Invalid model')


def r(x, zi, wi,model):
    return np.maximum(wi * hi(x, zi,model), 0)

def R(x, Z, W,model):
    m, n = Z.shape
    R = np.zeros(m)
    for i in range(R.size):
        R[i] = r(x, Z[i], W[i],model)
    return R

def f(x, Z, W,model):
    """
    returns value of the objective function
    """
    m, n = Z.shape
    return np.sum(R(x, Z, W,model)**2)

def dhi(x, zi,model):
    dh = np.zeros(x.size)
    A, c = make_Ac(x,2)
    if model == 1:
        dh[0] = zi[0]**2 - 2*zi[0]*c[0] + c[0]**2
        dh[1] = 2*(zi[0]*zi[1] - zi[0]*c[1] - zi[1]*c[0] + c[0]*c[1])
        dh[2] = zi[1]**2 - 2*zi[1]*c[1] + c[1]**2
        dh[3] = -2*zi[0]*A[0][0] + 2*c[0]*A[0][0] - 2*zi[1]*A[0][1] + 2*c[1]*A[0][1]
        dh[4] = -2*zi[0]*A[0][1] + 2*c[0]*A[0][1] - 2*zi[1]*A[1][1] + 2*c[1]*A[1][1]

    elif model == 2:
        dh[0] = zi[0]**2
        dh[1] = 2*zi[0]*zi[1]
        dh[2] = zi[1]**2
        dh[3] = -zi[0]
        dh[4] = -zi[1]

    else:
        raise ValueError('Invalid model')
    return dh


def dri(x, zi, wi,model):
    """
    returns gradient of the residual for a single point z = (x,y)
    """    
    ri = r(x, zi, wi,model)
    return (ri > 0) * dhi(x, zi,model) * wi

def jacobi(x, Z, W,model):
    """
    returns the jacobian of the residual-vector
    """
    m, n = Z.shape
    J = np.zeros((m, x.size))
    for i in range(m):
        J[i] = dri(x, Z[i], W[i],model)
    return J

def df(x, Z, W,model):
    """
    returns gradient of objective function
    """
    return 2 * (jacobi(x, Z, W,model).T).dot(R(x, Z, W,model))



def set_funcs(z,w,model):
    """
    returns anonymous functions for the objective function and its gradient
    """
    return (lambda x: f(x,z,w,model), lambda x: df(x,z,w,model))

if __name__ == "__main__":
    print('main')
    



    