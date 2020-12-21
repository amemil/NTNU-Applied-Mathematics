import numpy as np
from matplotlib import pyplot as plt


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

def generate_zw(z,A,c,N,d):
    """
    return array of labels and perturbed data-points
    """
    w1 = generate_w(z,A,c,N,d,model=1)
    w2 = generate_w(z,A,c,N,d,model=2)
    z_pert = z + np.random.uniform(low = -0.5,high = 0.5,size = (N,d))    

    return (z_pert,w1,w2)


##### Define and plot the hypersurface #####

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

def plot_surface(A,c,z,w,area,col,label,model):
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
    if not (label == 'gen' or label== '1st'):
        plt.scatter(z.T[0],z.T[1],c = colors)
    plt.grid(False)
############################################


if __name__ == "__main__":
    print('Main')


