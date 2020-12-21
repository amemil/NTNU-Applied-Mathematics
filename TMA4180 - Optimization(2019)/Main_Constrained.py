import numpy as np
from matplotlib import pyplot as plt
import Utilities_Constrained as util
#import test1 as tst
import algorithms_constrained as alg

np.random.seed(150)

# set the number of points
N,d = 100,2
scale = 2

plot_all = False

# set the model 
model = 2

constr_p = [0.5,20, 0] # [gamma_1, gamma_2, boundary for third constraint]

# set data generating hypersurface
A_gen = np.array([[1,-0.5], [-0.5,1]])  
c_gen = np.array([0,0]) 
  
# generate points and labels
x0,z = util.generate_random(N,d,scale)
z,w = util.generate_zw(z,A_gen,c_gen,N,d, model)

# set functions
FaugL,gradFaugL, L  = util.set_funcs_augL(z,w,model, constr_p)

# set initial conditions
x0 = np.array([2,0,2,0,0]) 
l0 = np.array([10,10,1])
mu0 = 100


# run the gradient descent algorithm 
#x1,it1, funcval1, l1, mu1, conv_p = alg.gradient_descent_augL(FaugL,gradFaugL,L, x0, 1e-12, mu0, l0)
#print('model = ', model, ' f = ', funcval1, ' k = ', it1)

#A_1,c_1 = util.make_Ac(x1,2)

#plt.clf()
#plt.figure(1)
#util.plot_surface(A_gen,c_gen,z,w,scale,'k','gen',model, '')
#util.plot_surface(A_1,c_1,z,w,scale,'g','augL',model, '')
#plt.show()


# Generate plot for model 1 and 2
plt.figure(2)
plt.subplot(121)
model=1
FaugL,gradFaugL, L  = util.set_funcs_augL(z,w,model, constr_p)
x1,it1, funcval1, l1, mu1, conv_p = alg.gradient_descent_augL(FaugL,gradFaugL,L, x0, 1e-12, mu0, l0)
A1, c1 = util.make_Ac(x1)
print('model = ', model, ' f = ', funcval1, ' k = ', it1)
util.plot_surface(A_gen,c_gen,z,w,scale,'k','gen',model,'')
util.plot_surface(A1,c1,z,w,scale,'g','augL1',model, 'Hypersurface corresponding with $x_{'+str(it1)+ '}$ and generating data')

plt.subplot(122)
model=2
FaugL,gradFaugL, L  = util.set_funcs_augL(z,w,model, constr_p)
x1,it1, funcval1, l1, mu1, conv_p = alg.gradient_descent_augL(FaugL,gradFaugL,L, x0, 1e-12, mu0, l0)
A1, c1 = util.make_Ac(x1)
print('model = ', model, ' f = ', funcval1, ' k = ', it1)
util.plot_surface(A_gen,c_gen,z,w,scale,'k','gen',model, '')
util.plot_surface(A1,c1,z,w,scale,'g','augL2',model, 'Hypersurface corresponding with $x_{'+str(it1)+'}$ and generating data')
plt.show()


# generate level plots
#plt.figure(3)
#fplot1 = lambda x, y: FaugL(np.array([x, x0[1], y, x0[3], x0[4]]), l0, mu0)
#fplot2 = lambda x, y: FaugL(np.array([x, y, x0[2], x0[3], x0[4]]), l0, mu0)
#fplot3 = lambda x, y: FaugL(np.array([x, x1[1], y, x1[3], x1[4]]), l1, mu1)
#fplot4 = lambda x, y: FaugL(np.array([x, y, x1[2], x1[3], x1[4]]), l1, mu1)
#plt.subplot(221)
#util.plot_level_curves(fplot1, [0.4, 2.5], [0.4, 2.5], 'contour', '$f_{augL}$ in function of $A_{11}$ and $A_{22}$ it=0', 'model='+str(model))
#plt.subplot(222)
#util.plot_level_curves(fplot2, [0.4, 4], [-3, 3], 'contour', '$f_{augL}$ in function of $A_{11}$ and $A_{12}$ it=0',  'model='+str(model))
#plt.subplot(223)
#util.plot_level_curves(fplot3, [0.4, 2.5], [0.4, 2.5], 'contour', '$f_{augL}$ in function of $A_{11}$ and $A_{22}$ it = '+str(it1),  'model='+str(model),conv_p[:,0], conv_p[:,2])
#plt.subplot(224)
#util.plot_level_curves(fplot4, [0.4, 4], [-3, 3], 'contour', '$f_{augL}$ in function of $A_{11}$ and $A_{12}$ it = '+str(it1),  'model='+str(model), conv_p[:,0], conv_p[:,1])
#plt.show()

