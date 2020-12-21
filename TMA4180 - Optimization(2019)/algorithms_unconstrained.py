import numpy as np
import numpy.linalg as la
import utilities_unconstrained as util
import test_unconstrained as tst


def bisection_linesearch(f,df,x,p,alpha0,c1,c2):
    """Bisection algorithm calculating a step length
        satisfying the Wolfe conditions
    """
    alpha = alpha0
    alpha_min = 0
    alpha_max = np.inf
    fx = f(x)
    dfx = df(x)
    dfxp = dfx.dot(p)
    
    Nmax = 100
    it = 0
    while alpha < 1e+10:
        if f(x + alpha*p) > fx + c1*alpha*dfxp and it < Nmax:
            # No sufficient decrease: Too short step
            alpha_max = alpha
            alpha = 0.5*(alpha_min + alpha_max)
            it +=1
        elif df(x + alpha*p).dot(p) < c2*dfxp and it < Nmax:
            # No curvature: Too short step
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha *= 2.0
                it+=1
            else:
                alpha = 0.5*(alpha_min + alpha_max)
                it+=1
        else:
            return alpha
    raise ValueError('Steplength is too long!')
    

def backtracking_linesearch(f,gradf,p,x):
    """
    Backtracking line search
    """
    Nmax = 1000
    c = 1e-4
    rho = 0.5
    alpha = 1
    
    f0 = f(x)
    phi = f(x+alpha*p)
    dF = gradf(x)
    
    it = 0
    while (phi > f0 + c*alpha*dF.dot(p) and it < Nmax):
        alpha *= rho
        phi = f(x+alpha*p)
        it += 1

    return alpha
  

def gradient_descent(f,grad,x0,tol):
    """
    Gradient descent algorithm
    """
    print('Gradient descent started')
    #c1,c2,alpha0 = 0.05, 0.5,1
    p_k = -grad(x0)
    x_k = x0
    
    Nmax = 10000
    
    it = 0
    while la.norm(grad(x_k)) > tol and it < Nmax:
        p_k /= la.norm(p_k)
        #alpha = bisection_linesearch(f,grad,x_k,p_k,alpha0,c1,c2)
        alpha = backtracking_linesearch(f,grad,p_k,x_k)
        x_k = x_k+alpha*p_k
        p_k = -grad(x_k)
        it += 1
        if alpha < 1e-17:
            print('No change, return iterate')
            return x_k,it,f(x_k)
        #print progress to console
        if it % 100 == 0:
            print('k =  {:}'.format(it))
            print("f(x_k) = {:}".format(f(x_k)))
            print('alpha_k = {:}'.format(alpha))
            print('')
        
    print('Gradient descent done')
    print('')
    return x_k,it,f(x_k)

def bfgs(f,grad,x0,z,w,tol):
    """
    bfgs algorithm
    """
    print('BFGS started')
    Nmax = 10000
    m,n = z.shape
    k = n*(n+1)//2
    alpha0,c1,c2 = 1,1e-4,0.5
    
    I = np.eye(n+k)
    H = I
    x_k = x0
    dF = grad(x_k)

    it = 0    
    while la.norm(dF) > tol and it < Nmax:        
        p_k = -H.dot(dF)
        p_k /= la.norm(p_k)
        
        alpha_k = bisection_linesearch(f,grad,x_k,p_k,alpha0,c1,c2)
        #alpha_k = backtracking_linesearch(f,grad,p_k,x_k)
        if alpha_k <1e-17:
            print('No change, return iterate')
            return x_k,it,f(x_k)
        x_next = x_k + alpha_k*p_k
        dF_next = grad(x_next)
        
        
        s_k = x_next - x_k
        y_k = dF_next - dF
        
        if (y_k.T@s_k <= 0 ):
            print('reboot')
            H = I
            continue
        
        rho_k = 1/(y_k.dot(s_k))
        
        H = (I-rho_k*s_k*y_k.T)@H@(I-rho_k*y_k*s_k.T) + rho_k*s_k*s_k.T
        it +=1
        
        #print progress to console, every 100th iteration if applicable
        if it % 100 == 0:
            print('k =  {:}'.format(it))
            print("f(x_k) = {:}".format(f(x_k)))
            print('alpha_k = {:}'.format(alpha_k))
            print('')
        
        x_k = x_next
        dF = dF_next
    print('BFGS done')
    print('')
    return x_k,it,f(x_k)


    
if __name__ == "__main__":
    print('main')
    
    
    

    