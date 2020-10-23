
import numpy as np              
import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy.stats import gamma
import random
from numba import njit
@njit

def learning_rule(s1,s2,Ap,Am,taup,taum,t,i,binsize): 
    '''
    s1,s2 : binary values for the different time bins for neuron 1 and 2 respectively, 1:spike, 0:no spike
    i : current iteration/timebin for the numerical approximation
    '''
    l = i - np.int(np.ceil(10*taup / binsize))
    return s2[i-1]*np.sum(s1[max([l,0]):i]*Ap*np.exp((t[max([l,0]):i]-max(t))/taup)) - s1[i-1]*np.sum(s2[max([l,0]):i]*Am*np.exp((t[max([l,0]):i]-max(t))/taum))

def logit(x):
    return np.log(x/(1-x))

def inverse_logit(x):
    return np.exp(x)/(1+np.exp(x))

class SimulatedData:
    '''
    Ap, Am, tau : learning rule parameters
    b1,b2 : background noise constants for neuron 1 and neuron 2, determnining their baseline firing rate
    w0 : start value for synapse strength between neuron 1 and 2. 
    '''
    def __init__(self,Ap=0.005, tau=0.02, std=0.001, sec=120.0, binsize=1/200.0,b1=-2.0, b2=-2.0, w0=1.0):
        self.Ap = Ap
        self.tau = tau
        self.std = std
        self.sec = sec
        self.binsize = binsize
        self.Am = 1.05*self.Ap
        self.b1 = b1
        self.b2 = b2
        self.w0 = w0
    
    def set_Ap(self,Ap):
        self.Ap = Ap
    def set_tau(self,tau):
        self.tau = tau
    def set_std(self,std):
        self.std = std
    def set_sec(self,sec):
        self.sec = sec
    def set_binsize(self,binsize):
        self.binsize = binsize
    def set_b1(self,b1):
        self.b1 = b1
    def set_b2(self,b2):
        self.b2 = b2
    def set_w0(self,w0):
        self.w0 = w0
    
    def get_Ap(self):
        return self.Ap
    def get_tau(self):
        return self.tau
    def get_std(self):
        return self.std
    def get_sec(self):
        return self.sec
    def get_binsize(self):
        return self.binsize
    def get_b1(self):
        return self.b1
    def get_b2(self):
        return self.b2
    def get_w0(self):
        return self.w0
        
    def get_dataset(self):
        iterations = np.int(self.sec/self.binsize)
        t,W,s1,s2 = np.zeros(iterations),np.zeros(iterations),np.zeros(iterations),np.zeros(iterations)
        W[0] = w0 #Initial value for weights
        s1[0] = np.random.binomial(1,inverse_logit(self.b1)) #5.4 in article, generate spike/not for neuron 1
        for i in tqdm(range(1,iterations)):
            lr = learning_rule(s1,s2,self.Ap,self.Am,self.tau,self.tau,t,i,self.binsize)
            W[i] = W[i-1] + lr + np.random.normal(0,self.std) #updating weights, as in 5.8 in article
            s2[i] = np.random.binomial(1,inverse_logit(W[i]*s1[i-1]+self.b2)) #5.5 in article, spike/not neuron 2
            s1[i] = np.random.binomial(1,inverse_logit(self.b1)) #5.4
            t[i] = self.binsize*i #list with times (start time of current bin)
        return(s1,s2,t,W)
        
    def plot_weight_trajectory(self,t,W):
        plt.figure()
        plt.title('Weight trajectory')
        plt.plot(t,W)
        plt.xlabel('Time')
        plt.ylabel('Weight')
        plt.show()
        
class ParameterInference:
    '''
    Class for estimating b1,b2,w0,Ap,Am,tau from SimulatedData
    '''
    def __init__(self, sec = 100, binsize = 1/200.0, P = 100, U = 100, it = 1500, N = 2):
        
