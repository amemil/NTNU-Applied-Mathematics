from matplotlib import pyplot as plt

import utilities_unconstrained as util
import algorithms_unconstrained as alg
import test_unconstrained as tst

import numpy as np

np.random.seed(150) 

N,d = 200,2
scale = 3
tol = 1e-6

#below code-chunk finds a random vector for generating the labels, but we
# restricted ourselves to the two test-cases described in the report.
"""
ellipse = True
x0,z = util.generate_random(N,d,scale)
A_start,c_start = util.make_Ac(x0,d)

 below code-snippet makes sure that A is positive definite -> ellipse
if ellipse:
    while util.not_posdef(A_start):
        x0,z = util.generate_random(N,d,scale)
        A_start,c_start = util.make_Ac(x0,d)
        print('not pos.def')
""" 

z = util.generate_random_z(N,d,scale)
x,y = z.T[0],z.T[1]

xmean = np.mean(x)
ymean = np.mean(y)



#A_gen = np.array([[-1,-3],[-3,-1]]) # HYPERBOLA
A_gen = np.array([[1,-0.5],[-0.5,1]]) # ELLIPSE

c_gen = np.array([xmean,ymean])

# perturb data, keep labels
z,w1,w2 = tst.generate_zw(z,A_gen,c_gen,N,d)


# starting vector
x0 = np.array([2,0,2,0,0])

# set functions
F_1,gradF_1 = util.set_funcs(z,w1,model = 1)
F_2,gradF_2 = util.set_funcs(z,w2,model = 2)

# call the algorithms
x1gd,it1gd, f1gd = alg.gradient_descent(F_1,gradF_1,x0,tol)
x1bfgs,it1bfgs,f1bfgs = alg.bfgs(F_1,gradF_1,x0,z,w1,tol)   

x2gd,it2gd,f2gd = alg.gradient_descent(F_2,gradF_2,x0,tol)
x2bfgs,it2bfgs,f2bfgs  = alg.bfgs(F_2,gradF_2,x0,z,w2,tol)

# construct matrices and vectors from the solution to plot hypersurfaces
A_1gd,c_1gd = util.make_Ac(x1gd,2)
A_1bfgs,c_1bfgs = util.make_Ac(x1bfgs,2)

A_2gd,c_2gd = util.make_Ac(x2gd,2)
A_2bfgs,c_2bfgs = util.make_Ac(x2bfgs,2)

# Plot solutions in 2x2 manner, where the rows correspons to model, and the
# columns corresponds to the methods
plt.figure(1)
plt.subplot(2,2,1)

tst.plot_surface(A_gen,c_gen,z,w1,scale,'k','gen',model=1)
tst.plot_surface(A_1gd,c_1gd,z,w1,scale,'g','GD',model=1)

plt.ylabel('Model 1')
plt.title('Gradient Descent')

plt.subplot(2,2,2)
tst.plot_surface(A_gen,c_gen,z,w1,scale,'k','gen',model=1)
tst.plot_surface(A_1bfgs,c_1bfgs,z,w1,scale,'g','BFGS',model=1)

plt.title('BFGS')


plt.subplot(2,2,3)
tst.plot_surface(A_gen,c_gen,z,w2,scale,'k','gen',model=2)
tst.plot_surface(A_2gd,c_2gd,z,w2,scale,'g','GD',model=2)

plt.ylabel('Model 2')

plt.subplot(2,2,4)
tst.plot_surface(A_gen,c_gen,z,w2,scale,'k','gen',model=2)
tst.plot_surface(A_2bfgs,c_2bfgs,z,w2,scale,'g','BFGS',model=2)




