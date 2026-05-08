import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import sys
import astropy
from astropy import units as u

### KONSTANTEN


k = (1.380649 * 10**(-23) * u.J/u.K).cgs
h = (6.626070 * 10**(-34) * u.J*u.s).cgs
me= (9.109384 * 10**(-31) * u.kg).cgs

ACa = 6.36 #Abundance of ca
Eion12 = (6.11316 * u.eV).cgs #Ca1 to Ca2
Eion23 = (11.87172 * u.eV).cgs #Ca2 to Ca3 #https://physics.nist.gov/PhysRefData/Handbook/Tables/calciumtable1.htm

saha_const = k*me*2*np.pi/(h**2)

###PARTITION FUNCTIONS


def pfCaI(T):
    
    a = np.array([-5.21521E+02, 4.10712446E+02, -1.26375869E+02, 1.90976557E+01, -1.42269952, 4.19085267E-02])
    lnZ = 0.0
    for i in range(0,6):
        lnZ += a[i] * (np.log(T.value)**i)
    #print(lnZ)
    
    return np.exp(lnZ)

def pfCaII(T):
    
    a = np.array([1.65874025E+03, -1.04392185E+03, 2.61702809E+02, -3.26369424E+01, 2.02350114, -4.98594460E-02])
    lnZ = 0.0
    for i in range(0,6):
        lnZ += a[i] * (np.log(T.value)**i)
    #print(lnZ)
    
    return np.exp(lnZ)

def pfCaIII(T):

    a = np.array([-1.24819814E-03, 7.75158073E-04, -1.92014703E-04, 2.37150053E-05, -1.46034751E-06, 3.58695382E-08])
    lnZ = 0.0
    for i in range(0,6):
        lnZ += a[i] * (np.log(T.value)**i)
    #print(lnZ)
    
    return np.exp(lnZ)

### CALCULATION OF CA2 BY USE OF SAHA EQUATIONS

# Saha: nj1 * ne/ nj = (2 * np.pi * me * k * T/h**2)**(3/2)  * 2*Zj1/Zj * np.exp(-Eion/k/T)


def GasEquation(P,T):
    # P = n * k * T
    n_total = P/(k*T)
    return  n_total / 1 # returns n

def N(n,ne):
    # n = nion + ne
    return n - ne
    
def NCa(ACa, nH):
    nCa = nH * 10.0 ** (ACa - 12.0)
    return nCa



def SahaEquation12(T): # from CaI to CaII
    Z1  =  pfCaI(T)
    Z_2 =  pfCaII(T)
    Saha12 = (saha_const*T)**(3/2)  * 2*Z_2/Z1 * np.exp(-Eion12/(k*T))
    return Saha12, Z1, Z_2 

def SahaEquation23(T): # from CaII to CaIII
    Z_3 = pfCaIII(T)
    Z2  = pfCaII(T)
    Saha23 = (saha_const*T)**(3/2)  * 2*Z_3/Z2 * np.exp(-Eion23/(k*T))
    return Saha23, Z2, Z_3  


    
def NCa2(nCa, T, ne, verbose = False):
    #print('Hi I am in NCa2 function!')
    Saha12, Z1, Z_2  = SahaEquation12(T)
    Saha23, Z2, Z_3  = SahaEquation23(T)
    nCaII = nCa/(ne/Saha23 + 1 + Saha12/ne)
    return nCaII,  Saha12, Z1, Z_2,  Saha23, Z2, Z_3


def numberdensCa2(P,T,ne, verbose = False):

    n = GasEquation(P,T)
    if verbose:
        print("info::numbedensCa2: total number density: ", n)
    
    nion = N(n,ne)
    if verbose:
        print("info::numbedensCa2: total number density of all the ions: ", nion)
    
    nCa = NCa(ACa, n)
    if verbose:
        print("info::numbedensCa2: total number density of Ca: ", nCa)
    

    nCaII,  Saha12, Z1, Z_2,  Saha23, Z2, Z_3 = NCa2(nCa,T, ne, True)
 
    if verbose:
        print(f"info::numbedensCa2: total number density of Ca II: {nCaII}")
    
    return nCaII,  Saha12, Z1, Z_2,  Saha23, Z2, Z_3

