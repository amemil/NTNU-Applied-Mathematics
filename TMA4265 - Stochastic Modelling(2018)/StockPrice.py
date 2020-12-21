import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

sigma = 0.75
mu = 0
############################ question 1 ####################################


#ANALYTICAL:
sigmatot = np.sqrt(120)*sigma
mushift = 40
aprob = norm.sf(50, loc=mushift, scale=sigmatot)

#SIMULATION:

x = [] #all realizations
t = np.linspace(0,120,121)
realizations = 100

for i in range(realizations):
    xreal = [40] #x0 = 40
    z = np.random.normal(size=len(t)-1)
    for i in range(len(t)-1):
        xreal.append(xreal[i] + np.sqrt(t[i+1]-t[i])*sigma*z[i])
    x.append(xreal)


count50 = 0
plt.figure(1)
for i in range(realizations):
    plt.plot(t,x[i])
    if x[i][-1] > 50:
        count50 += 1

print(count50)
print(aprob)
plt.plot(np.linspace(115,120,6),np.linspace(50,50,6),'k-',label='Amount of prizes greater than 50$: '+str(count50))
plt.plot(np.linspace(115,115,6),np.linspace(50,70,6),'k-')
plt.plot(np.linspace(115,120,6),np.linspace(70,70,6),'k-')
plt.title(str(realizations)+' realizations of the Brownian motion for stock prize')
plt.ylabel('Stock prize [$]')
plt.xlabel('Time [days after Jan 1st]')
plt.legend()
plt.show()

    
############################ question 2 ####################################

#ANALYTICAL:
sigmatot2 = np.sqrt(60)*sigma
mushift2 = 45
aprob2 = norm.sf(50, loc=mushift2, scale=sigmatot2)

#NUMERICAL: 

x2 = [] #all realizations
t2 = np.linspace(60,120,61)
realizations = 100

for i in range(realizations):
    xreal2 = [45] #x(60) = 45
    z2 = np.random.normal(size=len(t2)-1)
    for i in range(len(t2)-1):
        xreal2.append(xreal2[i] + np.sqrt(t2[i+1]-t2[i])*sigma*z2[i])
    x2.append(xreal2)


count50_2 = 0
plt.figure(2)
for i in range(realizations):
    plt.plot(t2,x2[i])
    if x2[i][-1] > 50:
        count50_2 += 1

print(count50_2)
print(aprob2)
plt.title(str(realizations)+' realizations of the Brownian motion for stock prize')
plt.plot(np.linspace(115,120,6),np.linspace(50,50,6),'k-',label='Amount of prizes greater than 50$: '+str(count50_2))
plt.plot(np.linspace(115,115,6),np.linspace(50,70,6),'k-')
plt.plot(np.linspace(115,120,6),np.linspace(70,70,6),'k-')
plt.ylabel('Stock prize [$]')
plt.xlabel('Time [days after Jan 1st]')
plt.legend()
plt.show()

############################ question 3 ####################################

#ANALYTICAL:
a = 4
def cumulativhit(t,sigma,a): #cumulative distribution for hitting times
    return 2*(1-norm.cdf(a/(np.sqrt(t*sigma**2))))
    
def marginalhit(t,sigma,a): #probability density function
    return (a/(np.sqrt(2*sigma**2*t**3*math.pi)))*np.exp(-(a**2)/(2*t*sigma**2)) 

distribution = []
marginaldist = []
for i in range(1,10001):
    distribution.append(cumulativhit(i,sigma,a))
    marginaldist.append(marginalhit(i,sigma,a,mu))


realizations = 10000
x3 = []
t3 = []
hitting_times_marg = np.linspace(0,0,10000)
hitting_times_cum = np.linspace(0,0,10000)

for i in range(realizations):
    print(i)
    xreal3 = [40] #x(0) = 40
    treal3 = [0]
    z3 = np.random.normal()
    while xreal3[-1] < 44 and treal3[-1] < 10000:
        treal3.append(treal3[-1]+1)
        xreal3.append(xreal3[-1] + np.sqrt(treal3[-1]-treal3[-2])*sigma*z3)
        z3 = np.random.normal()
    x3.append(xreal3)
    t3.append(treal3)
    if treal3[-1] != 10000:
        for j in range(treal3[-1] - 1,10000):
            hitting_times_cum[j] += 1
        hitting_times_marg[treal3[-1] - 1] += 1


                                        # PLOTTING Question 3 # 
plt.figure(3)
for i in range(realizations):
    plt.plot(t3[i],x3[i])

plt.show()

fig, ax1 = plt.subplots(figsize=(6.4,5.2))
color = 'tab:red'
ax1.set_xlabel('Time [t]')
ax1.set_ylabel('Analytical probability $P(T_{a} \leq t)$', color=color)
ax1.plot(np.linspace(1,10000,10000),distribution, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Realizations', color=color)  # we already handled the x-label with ax1
ax2.plot(np.linspace(1,10000,10000),hitting_times_cum, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.subplots_adjust(top=0.92)
plt.title('Analytical and empirical probabilities')
#plt.savefig('stockc')
plt.show()

fig, ax1 = plt.subplots(figsize=(6.4,5.2))
color = 'tab:red'
ax1.set_xlabel('Time [t]')
ax1.set_ylabel('Analytical probability $P(T_{a} = t)$', color=color)
ax1.plot(np.linspace(1,10000,10000),marginaldist, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Realizations', color=color)  # we already handled the x-label with ax1
ax2.plot(np.linspace(1,10000,10000),hitting_times_marg, 'bx',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.subplots_adjust(top=0.92)
plt.title('Analytical and empirical probabilities')
#plt.savefig('stockc')
plt.show()

