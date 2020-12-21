from sympy import Symbol, nsolve
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

maudfont = {'fontname':'Times New Roman', 'size': 16}
mpl.rc('font',family='Times New Roman')

x = Symbol('x')
y = Symbol('y')


eps_a = 0.875
sigma = 5.6703*10**(-8)
f = 0.618
eps_e = 1 
cc = 0.66
rlc = 0.195
alpha = 3
beta = 4
rse = 0.17
ps0 = 341.3
asw = 0.1451
asc = 0.1239
rsm = 0.1065
a03 = 0.08
alc = 0.622
rsc = 0.22
alw = 0.8258 
dikt = {'cc':cc, 'rlc':rlc,  'rse':rse, 'asw':asw, 'asc':asc, 'rsm':rsm, 'a03':a03, 'alc':alc, 'rsc':rsc, 
        'alw':alw, 'eps_a':eps_a, 'sigma':sigma, 'eps_e':eps_e, 'ps0':ps0, 'f':f, 'alpha':alpha, 'beta':beta}


def KtoC(K):
    return K - 273.15

def calculateTemp(cc, rlc, rse, asw, asc, rsm, a03, alc, rsc, alw, eps_a, sigma, eps_e, ps0, f, alpha, beta):
    
    #atmosfære-koeffs
    refl_sw = (1-cc)*rsm + cc*rsc  # vektet refl for mol og sky
    abs_sw = ((1-cc)*(0.5*asw + 0.5*a03) + cc*asc)

    refl_lw = cc*rlc
    abs_lw = ((1-cc)*alw + cc*alc)  # vektet abs for molekyl og sky ganger

    #---------------Ledd for likning 1---------------------------#
    Pae = eps_a * sigma * x**4
    Pea_refl = refl_lw * eps_e * sigma * y**4
    # Psa_trans = ps0*(1-(asw+rsm))*(1-(a03+rsm))*(1-cc*(asc+rsc)) #3 -lags
    Psa_trans = (1-abs_sw) * (1-refl_sw) * ps0

    Pse_refl = Psa_trans * rse
    Pea_em = eps_e * sigma * y**4
    Pea_hl = -(alpha+beta)*(x-y)
    Pse_refl2 = Pse_refl * refl_sw

    # Equation 1
    eq1 = Pae + Pea_refl + Psa_trans - Pse_refl - Pea_hl - Pea_em + Pse_refl2  

    # Ledd for likning 2
    Ps0 = ps0
    Psa_refl = ps0 * refl_sw
    Pea_trans = Pea_em*(1-abs_lw)*(1-refl_lw)
    Pas = f*eps_a*sigma*x**4
    Pse_refl_trans = Pse_refl * (1-abs_sw) * (1-refl_sw)

    # Equation 2
    eq2 = - Ps0 + Psa_refl + Pea_trans + Pas +  Pse_refl_trans

    los = nsolve((eq1, eq2), (x, y), (320, 350))
    return [los[0],los[1]]

temp = calculateTemp(**dikt)


#generere dataframe med T-verdier basert på prosent endring 
def sensitivityCalculation():
    T=np.zeros((len(dikt)+1,2))
    T[0] = calculateTemp(**dikt)
    for i in range(17):
        temp = [*dikt.values()]
        temp[i] *= 1.01
        T[i+1] = calculateTemp(*temp)
    T=T.round(3)
    df = pd.DataFrame(data=T, index=['original',*dikt.keys()], columns=[r'$T_A$',r'$T_E$'])
    df.drop(labels=['ps0', 'sigma', 'cc', 'rsc', 'asc', 'rlc', 'alc', 'eps_e', 'eps_a', 'f', 'alpha', 'beta'],axis=0, inplace=True)
    return df

sensitivityCalculation()

#plot sensitivitetsanalyse
def sensitivityBarPlot():
    norm = sensitivityCalculation()
    norm[r'$T_A$'] = norm[r'$T_A$'] / norm[r'$T_A$']['original']
    norm[r'$T_E$'] = norm[r'$T_E$'] / norm[r'$T_E$']['original']
    norm = norm.rename(index={'original':r'original', 'rse':r'$r_{SE}$', 'asw':r'$\alpha_{SE}$', 'rsm':r'$r_{SM}$', 'a03':r'$\alpha_{O_3}$', 'alw':r'$\alpha_{LW}$'})
    plt.figure()
    norm.plot(kind='bar',ylim=[0.998,1.002])
    plt.title('Sensitivity Analysis',**maudfont)
    plt.xlabel('Quantity Changed',**maudfont)
    plt.ylabel('Relative Change in Temperature',**maudfont)
    plt.axhline(y=1.0, color='black', linestyle='--')
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('sensitivity.pdf')

sensitivityBarPlot()


names = ['Cc','$r_{LC}$','$r_{SE}$','$a_{SW}$','$a_{SW}$','$r_{SM}$','$a_{03}$','$a_{LC}$','$r_{SC}$','$a_{LW}$']
nameses = ['Cc','r_LC','r_SE','a_SW','a_SC','r_SM','a_03','a_LC','r_SC','a_LW']


def sensitivity(TrueValues,names,eps_e,eps_a,ps0,sigma,f,alpha,beta):
    for i in range(len(TrueValues)):
        TempEarth = []
        TempAtmos = []
        tempValues = np.copy(TrueValues)
        values = np.linspace(0,1,100)
        if i == 5:
            values = np.linspace(0,0.945,100)
        for j in range(len(values)):
            tempValues[i] = values[j]
            TempEarth.append(KtoC(calculateTemp(*tempValues,eps_a,sigma,eps_e,ps0,f,alpha,beta)[1]))
            TempAtmos.append(KtoC(calculateTemp(*tempValues,eps_a,sigma,eps_e,ps0,f,alpha,beta)[0]))
        plt.figure()
        plt.title('Sensitivity of '+names[i],**maudfont)
        plt.xlabel(names[i]+' value',**maudfont)
        plt.ylabel('Temperature (Celcius)',**maudfont)
        plt.plot(values,TempEarth,label='Earth')
        plt.plot(values,TempAtmos,label='Atmosphere')
        plt.axvline(TrueValues[i],color='g',linestyle='--',label='True Value')
        plt.legend()
        plt.savefig(nameses[i]+'.pdf')
        plt.show()
        
        
TrueValues = np.array([cc,rlc,rse,asw,asc,rsm,a03,alc,rsc,alw])

earthTempsKelvin = calculateTemp(cc, rlc, rse, asw, asc, rsm, a03, alc, rsc, alw, eps_a, sigma, eps_e, ps0, f, alpha, beta)
print(KtoC(earthTempsKelvin[0]),"degrees and",KtoC(earthTempsKelvin[1]),"degrees")
    
sensitivity(TrueValues,names,eps_e,eps_a,ps0,sigma,f,alpha,beta)


