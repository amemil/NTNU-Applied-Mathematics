import numpy as np
import numpy.linalg as la
import algorithms as alg
import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
def r_i(c, r, z_i):
    return la.norm(z_i-c, 2)**2 -r**2

def R(c, r, z):
    res = np.zeros((len(z),1))
    for i in range(0, len(z)):
        res[i] = r_i(c, r, z[i])
    return res
        
def dRdc(c, r, z):
    res = np.zeros(z.shape)
    for i in range(0, len(z)):
        res[i,:] = -2*(z[i]-c)
    return res

def gen_data(xrange, yrange, c1, c2, r1, r2, N, per):
    points_x = (np.random.rand(N)-0.5)*xrange
    points_y = (np.random.rand(N)-0.5)*yrange
    
    w = np.array([points_x, points_y])
    w = w.T
    z = np.zeros((len(w),1))
    for i in range(0, len(w)):
        if r_i(c1, r1, w[i]) < 0:
            z[i] = -1 # circle 1
        if r_i(c2, r2, w[i]) < 0:
            z[i] = 1# cirlce 2
    
    for point in w:
        perturbation = np.random.randn(2)*per
        point += perturbation
    return w, z

def plot_result(label, xrange, yrange, w, z, c1=None, r1=None, c2=None, r2=None, N=10, color='b', title=''):
    colors = ['g' if i==-1 else 'r' if i==1 else 'b' for i in z]
    delta = 1.0/N
    plt.scatter(w[:,0], w[:,1], color=colors)
    if r1 is not None:
        fplot = lambda z: r_i(c1, r1, z)
        xx = np.arange(-xrange*0.5, xrange*0.5 , delta)
        yy = np.arange(-yrange*0.5, yrange*0.5, delta)
        X, Y = np.meshgrid(xx, yy)
        Z = []
        for y_i in range(0,len(yy)):
            Z += [[]]
            for x_i in range(0,len(xx)):
                Z[y_i]  += [fplot(np.array([xx[x_i],yy[y_i]]))]
        CS = plt.contour(X, Y, Z, [0], linewidths=2, zorder=10, color = color)
        plt.clabel(CS, fmt='{0}1'.format(label), fontsize=10)
    if r2 is not None:
        fplot = lambda z: r_i(c2, r2, z)
        xx = np.arange(-xrange*0.5, xrange*0.5 , delta)
        yy = np.arange(-yrange*0.5, yrange*0.5, delta)
        X, Y = np.meshgrid(xx, yy)
        Z = []
        for y_i in range(0,len(yy)):
            Z += [[]]
            for x_i in range(0,len(xx)):
                Z[y_i]  += [fplot(np.array([xx[x_i],yy[y_i]]))]
        CS = plt.contour(X, Y, Z, [0], linewidths=2, zorder=10, color= color)
        plt.clabel(CS, fmt='{0}2'.format(label), fontsize=10)
    plt.title(title)
    plt.axis('equal')
    #plt.show()
        
def x2circles(x):
    return np.array(x[0:2]), x[2], np.array(x[3:5]), x[5]

def circles2x(c1, r1, c2, r2):
    x = [0,0,0,0,0,0]
    x[0:2] = c1
    x[2] = r1
    x[3:5] = c2
    x[5] = r2
    return np.array(x)

def f3(x, w, z):
    c1, r1, c2, r2 = x2circles(x)
    label1datx = np.extract(z==-1,w[:,0])
    label1daty = np.extract(z==-1, w[:,1])
    label1dat =np.array([label1datx, label1daty]).T
    
    label2datx = np.extract(z==1,w[:,0])
    label2daty = np.extract(z==1, w[:,1])
    label2dat =np.array([label2datx, label2daty]).T
    
    labeln1datx = np.extract(z!=-1,w[:,0])
    labeln1daty = np.extract(z!=-1, w[:,1])
    labeln1dat =np.array([labeln1datx, labeln1daty]).T
    
    labeln2datx = np.extract(z!=1,w[:,0])
    labeln2daty = np.extract(z!=1, w[:,1])
    labeln2dat =np.array([labeln2datx, labeln2daty]).T
    
    #plt.scatter(labeln1dat[:,0], labeln1dat[:,1], color='orange')
    #plt.scatter(labeln2dat[:,0], labeln2dat[:,1], color='black')
    
    if len(label1datx) != 0:
        p1 = np.sum(np.max(R(c1,r1,label1dat),0)**2)
    else:
        p1 = 0
    p2 = np.sum(np.max(R(c2,r2,label2dat),0)**2)
    p3 = np.sum(np.min(R(c1,r1,labeln1dat),0)**2)
    p4 = np.sum(np.min(R(c2,r2,labeln2dat),0)**2)
        
    return p1+p2+p3+p4

def df3(x, w, z):
    c1, r1, c2, r2 = x2circles(x)
     
    Rc1 = R(c1,r1, w)
    Rc2 = R(c2,r2, w)
    
    label1datx = np.extract(np.logical_and(z==-1,Rc1 >0),w[:,0])
    label1daty = np.extract(np.logical_and(z==-1,Rc1 >0), w[:,1])
    label1dat =np.array([label1datx, label1daty]).T
    
    label2datx = np.extract(np.logical_and(z==1 , Rc2 > 0),w[:,0])
    label2daty = np.extract(np.logical_and(z==1 , Rc2 > 0), w[:,1])
    label2dat =np.array([label2datx, label2daty]).T
    
    labeln1datx = np.extract(np.logical_and(z!=-1 , Rc1<0),w[:,0])
    labeln1daty = np.extract(np.logical_and(z!=-1 , Rc1<0), w[:,1])
    labeln1dat =np.array([labeln1datx, labeln1daty]).T
    
    labeln2datx = np.extract(np.logical_and(z!=1 , Rc2<0),w[:,0])
    labeln2daty = np.extract(np.logical_and(z!=1 , Rc2<0), w[:,1])
    labeln2dat =np.array([labeln2datx, labeln2daty]).T
    
    dp1dc1 = 2*R(c1,r1, label1dat).T.dot(dRdc(c1,r1, label1dat))
    dp1dr1 = np.sum(-4*R(c1,r1, label1dat)*r1)
    
    dp2dc1 = 2*R(c1,r1, labeln1dat).T.dot(dRdc(c1,r1, labeln1dat))
    dp2dr1 = np.sum(-4*R(c1,r1, labeln1dat)*r1)

    #plt.scatter(labeln1datx, labeln1daty, color='orange')
    #plt.scatter(labeln2datx, labeln2daty, color='purple')
    #plt.show()


    dp3dc2 = 2*R(c2,r2, label2dat).T.dot(dRdc(c2,r2, label2dat))
    dp3dr2 = np.sum(-4*R(c2,r2, label2dat)*r2)
    


    dp4dr2 = np.sum(-4*R(c2,r2, labeln2dat)*r2)
    dp4dc2 = 2*R(c2,r2, labeln2dat).T.dot(dRdc(c2,r2, labeln2dat))

    
    dc1 = dp1dc1+dp2dc1
    dc2 = dp3dc2+dp4dc2
    dr1 = dp1dr1 + dp2dr1
    dr2 = dp3dr2 + dp4dr2
    
    
    dx = circles2x(dc1[0], dr1, dc2[0], dr2)
    
    return dx 

def co0(x):
    c1, r1, c2, r2 = x2circles(x)
    if -r1<0:
        return 0.0
    else:
        return -r1
    
def dco0(x):
    c1, r1, c2, r2 = x2circles(x)
    if co0(x)>0:
        return np.array([0,0, -1, 0, 0, 0])
    else: 
        return np.array([0,0,0,0,0,0])

def co1(x):
    c1, r1, c2, r2 = x2circles(x)
    if -r2<0:
        return 0.0
    else:
        return -r2
    
def dco1(x):
    c1, r1, c2, r2 = x2circles(x)
    if co1(x)>0:
        return np.array([0,0,0,0,0,-1])
    else:
        return np.array([0,0,0,0,0,0])
    
def co2(x):
    c1, r1, c2, r2 = x2circles(x)
    res = -la.norm(c1-c2)+abs(r1)+abs(r2)
    if res <0:
        return 0.0
    else:
        return res
    
def dco2(x):
    c1, r1, c2, r2 = x2circles(x)
    if co2(x)>0:
        dc2dr1=r1
        dc2dr2=r2
        dc2dc1 = -0.5/la.norm(c1-c2)*2*(c1-c2)
        dc2dc2 = 0.5/la.norm(c1-c2)*2*(c1-c2)
        return circles2x(dc2dc1, dc2dr1, dc2dc2, dc2dr2)
    else:
        return np.array([0,0,0,0,0,0])
    
def f3augL(x, w, z, l, mu):
    p1 = f3(x, w, z)
    p2 = -l[0]*co0(x)-l[1]*co1(x)-l[2]*co2(x)
    p3 = 0.5*mu**2*(co0(x)**2+co1(x)**2+co2(x)**2)
    return p1+p2+p3

def df3augL(x, w, z, l, mu):
    p1 = df3(x, w, z)
    p2 = -l[0]*dco0(x)-l[1]*dco1(x)-l[2]*dco2(x)
    p3 = mu*(co0(x)*dco0(x)+co1(x)*dco1(x)+co2(x)*dco2(x))
    return p1+p2+p3


def set_funcs_augL(z,w):
    return lambda x, l, mu: f3augL(x, w, z, l, mu), \
           lambda x, l, mu: df3augL(x, w, z, l, mu),\
           lambda l, x, mu: updateL(l, x, mu ), 
        
def generate_w(z,A,c,N,d,model):
    x_sample = z.T[0]
    y_sample = z.T[1]

    eps = 1e-10
    labels = -np.ones(len(x_sample))
    if model == 1:
        for i in range(len(x_sample)):
                x = (x_sample[i],y_sample[i])
                Ax = A@(x-c)
                if (x-c).T@Ax -1.0 < eps:
                    labels[i] = 1
    elif model == 2:
        b = c
        for i in range(len(x_sample)):
            z = np.array([x_sample[i],y_sample[i]])
            Az = A@z
            if z.T@Az- z.T@b -1.0 < eps:
                labels[i] = 1
    else:
        raise ValueError('Invalid model choice')
    return labels

def generate_zw(z,A,c,N,d,model):
    """
    return array of labels and perturbed data-points
    """
    w = generate_w(z,A,c,N,d,model)

    #perturb datapoints - keep labels constant
    z_pert = z + np.random.uniform(low = -0.5,high = 0.5,size = (N,d))    

    return (z_pert,w)



def make_Ac(x,d =2):
    A = np.zeros((d,d))
    c = np.zeros(d)
    
    A[0][0] = x[0]
    A[0][1],A[1][0] = x[1],x[1]
    A[1][1] = x[2]
    
    c[0] = x[3]
    c[1] = x[4]
    return A, c

def not_posdef(A):
    """
    checks whether any 2x2 matrix A is not positive definite
    """
    return ((A[0][0] <= 0 and A[1][1] <= 0 ) or la.det(A) <= 0)

def generate_random(N,d,scale):
    x = np.random.randn(5)
    z = np.random.uniform(low = -scale,high=scale,size=(N,d)) # randn
    return x,z 

def hi(x, zi,model):
    A, c = make_Ac(x,2)
    if model == 1:
        return (zi-c).dot(A.dot(zi-c)) - 1
    elif model == 2:
        b = c
        return zi.T@A@zi - zi.T@b - 1
    else:
        raise ValueError('Invalid model')

# Calculate residual r for single point z
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


############# Objective function #########################################

def faugL(x,l,mu, Z,W, model, constr_p):
    p1 = f(x,Z,W, model)
    p2 = -l[0]*c0(x, constr_p) - l[1]*c1(x, constr_p) - l[2]*c2(x, constr_p)
    p3 = mu*0.5*(c0(x,constr_p)**2 + c1(x,constr_p)**2 + c2(x,constr_p)**2 )
    #print('c0=', c0(x, constr_p), 'c1=', c1(x, constr_p), 'c2=', c2(x, constr_p))
    return  p2 + p3  + p1

def dfaugL(x, l,mu,Z, W,model, constr_p):
    """
    returns gradient of objective function
    """
    dfdx = 2 * (jacobi(x, Z, W,model).T).dot(R(x, Z, W,model))
    dc0dx = l[0]*dc0(x, constr_p)
    dc1dx = l[1]*dc1(x, constr_p)
    dc2dx = l[2]*dc2(x, constr_p)
    
    ret = dfdx - dc0dx -dc1dx - dc2dx \
    +mu*(c1(x, constr_p)*dc1dx+c2(x, constr_p)*dc2dx+c0(x, constr_p)*dc0dx)
    return ret

############## Constraints ##############################################@
    
def c0(x, constr_p):
    A, c = make_Ac(x)
    ret = (A[0,0]-constr_p[0])*(A[0,0]-constr_p[1])
    if ret<0:
        return 0.0
    return ret

def c1(x, constr_p):
    A, c = make_Ac(x)
    ret= (A[1,1]-constr_p[0])*(A[1,1]-constr_p[1])
    if ret<0:
        return 0.0
    return ret

def c2(x, constr_p):
    A, c = make_Ac(x)
    ret= -np.sqrt(A[0,0]*A[1,1])+np.sqrt(constr_p[0]**2+A[0,1]**2)+constr_p[2]
    if ret<0:
        return 0.0
    return ret

def dc0(x, constr_p):
    if c0(x,constr_p)<=0.0:
        return 0.0
    else:
        A, c = make_Ac(x)
        dc1dA00 = 2*A[0,0]-constr_p[0]-constr_p[1]
    return np.array([dc1dA00, 0, 0, 0, 0])
    
def dc1(x, constr_p):
    if c1(x, constr_p)<=0.0:
        return 0.0
    else:
        A, c = make_Ac(x)
        dc1dA11 = 2*A[1,1]-constr_p[0]-constr_p[1]
        return np.array([0, 0, dc1dA11, 0, 0])
def dc2(x, constr_p):
    if c2(x, constr_p)<=0.0:
        return 0.0
    else:
        A, c = make_Ac(x)
        dc2dA00=-0.5/np.sqrt(A[0,0]*A[1,1])*A[1,1]
        dc2dA11=-0.5/np.sqrt(A[0,0]*A[1,1])*A[0,0]
        dc2dA12=(constr_p[0]**2+A[0,1]**2)*A[0,1]
    return np.array([dc2dA00, dc2dA12, dc2dA11, 0, 0])

def updateL(l, x, mu, constr_p):
    l[0] = l[0]- mu*c0(x,constr_p)
    l[1] = l[1]- mu*c1(x,constr_p)
    l[2] = l[2] -mu*c2(x,constr_p)
    return l




############ Augmented lagrangian for constrained setting ###################

def set_funcs_augL(z,w,model, constr_p):
    """
    returns anonymous functions for the objective function and its gradient
    """
    return lambda x, l, mu: faugL(x,l,mu, z,w, model, constr_p), \
           lambda x, l, mu: dfaugL(x, l,mu,z, w,model, constr_p),\
           lambda l, x, mu: updateL(l, x, mu, constr_p ), \
  

########## plot ##################################################
def plot_level_curves(
    f, 
    xrange, 
    yrange, 
    mode, 
    title, 
    param,  
    xpoints=None, 
    ypoints=None, 
    A1 = None, 
    c1 = None,
    As = None, 
    cs = None, 
    delta = 0.05
    ):
    
    
    x_lower = xrange[0]
    x_upper = xrange[1]
    y_lower = yrange[0]
    y_upper = yrange[1]
    xx = np.arange(x_lower,x_upper,delta)
    yy = np.arange(y_lower,y_upper,delta)
    X, Y = np.meshgrid(xx,yy)
    Z = []
    now = datetime.datetime.now()
    for y_i in range(0, len(yy)):
        Z +=  [[]]
        for x_i in range(0,len(xx)):
            Z[y_i].append(f(xx[x_i], yy[y_i]))
    Z = np.array(Z)
    if mode == 'contour':                                    
        plt.imshow(Z, interpolation='bilinear',
                        origin='lower', cmap=cm.Blues, 
                        extent=(x_lower,x_upper,y_lower,y_upper))
        plt.contour(X,Y,Z, cmap =cm.Blues)
        
    if mode == '3D':
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface( X, Y, Z)
    try:
        if xpoints.all()!= None:
            plt.plot(xpoints,ypoints,'k-',linewidth=1)
    except: 
        pass
    
    #plt.show()
    plt.savefig('level_curves/{0}_{1}_{2}{3}{4}_{5}{6}{7}.png'.format(title, param, now.year, now.month, now.day, now.hour, now.minute, now.second))
    plt.title(title)
    
    try:
        if A1.all()!=None:
            tst.plot_surface(As,cs,z,w,scale,'k','gen',model)
            tst.plot_surface(A1,c1,z,w,scale,'g','augL',model)
            plt.show()
            plt.savefig('results/{0}_{1}_{2}{3}{4}_{5}{6}{7}.png'.format(title, param, now.year, now.month, now.day, now.hour, now.minute, now.second))
    except:
        pass

def surface(X,Y,A,c,model):
    """
    returns the hypersurface
    """
    if model == 1:
        return A[0][0]*(X-c[0])**2+2*A[0][1]*(X-c[0])*(Y-c[1])+A[1][1]*(Y-c[1])**2-1
    elif model == 2:
        b = c
        return A[0][0]*X**2 + 2*A[0][1]*X*Y + A[1][1]*Y**2 - (b[0]*X + b[1]*Y) -1
    else:
        raise ValueError('Invalid model choice')

def plot_surface(A,c,z,w,area,col,label,model, title):
    """
    Plots hypersurface. 
    If ellipse, also plot the ellipse seperating the data
    """
    delta = 0.01
    colors = ['r' if i == 1 else 'b' for i in w]
    x = np.arange(-area*1.2, 1.2*area+delta, delta)
    y = np.arange(-area*1.2, 1.2*area+delta, delta)
    X, Y = np.meshgrid(x, y)
    Z = surface(X, Y, A, c,model)
    CS = plt.contour(X, Y, Z, [0], linewidths=2, zorder=10, colors = col)
    plt.clabel(CS, fmt=label, fontsize=10)
    if not label == 'gen':
        plt.scatter(z.T[0],z.T[1],c = colors)
    plt.grid(False)
    plt.title(title)

if __name__ == "__main__":
    print('run main instead of utilities to see results')
    
    
    


    