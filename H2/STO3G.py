# paqueterías
import numpy as np
from numpy.typing import NDArray # type annotation
from scipy.special import factorial2 # doble factorial
from scipy.special import sph_harm_y as sph_harm # armónicos esféricos

import sys

# funciones personalizadas para graficar
from graficar import plt, compararSTO

#############################################
########## COEFS y EXPONENTES
#############################################

# Stewart
dSte: list[float] = [0.154329, 0.535328, 0.444635]
aSte: list[float] = [2.22766, 0.405771, 0.109818]

# Szabo & Ostlund
dSO: NDArray = np.array([0.444635, 0.535328, 0.154329])
aSO: NDArray = np.array([0.168856, 0.623913, 3.42525])

# Este trabajo
data: NDArray = np.loadtxt("data/STO3G.csv", delimiter=",", usecols=(0,1), dtype=('str')) # cargar valores calculados
valores = dict()
for valor in data:
    valores[str(valor[0])] = float(valor[1])
    
d: NDArray = np.array([valores['d1'], valores['d2'], valores['d3']]) # coeficientes de contracción
a1: NDArray = np.array([valores['a1'], valores['a2'], valores['a3']]) # exponentes orbitales Gaussianos (Z=1)
a2: NDArray = np.array([valores['a1_2'], valores['a2_2'], valores['a3_2']]) # exponentes orbitales Gaussianos (Z=1.24)

# distancia interatómica de 1.4 a0 (radios de Bohr)
RA: NDArray = np.array([0, 0, 0])
RB: NDArray = np.array([1.4, 0, 0])

# carga nuclear
ZA: float = 1.0
ZB: float = 1.0

#############################################
########## FUNCIONES
#############################################

def R_STO(n: int, zeta: float, r: NDArray) -> NDArray:
    """ Función radial de tipo Slater

    Parámetros
        n : número cuántico principal
        zeta : exponente oribital de Slater
        r : coordenada radial
    """
    return np.power(2*zeta, 1.5) / np.sqrt(factorial2(2*n)) * np.power(2*zeta*r, n-1) * np.exp(-zeta*r)

def STO(n: int, l: int, m: int, zeta: float, r: NDArray, theta: float, phi: float) -> NDArray:
    """ Función de tipo Slater (Slater Type Orbital)

    Parámetros
        (n,l,m) : número cuántico (principal, azimutal, magnético)
        zeta : exponente orbital de Slater
        (r,theta,phi) : coordenada (radial, polar, azimutal)
    """
    return R_STO(n, zeta, r) * sph_harm(l, m, phi, theta)

def R_GTO(l: int, a: float, r: NDArray) -> NDArray:
    """ Función radial de tipo Guassiana

    Parámetros
        n : número cuántico principal
        a : exponente orbital Gaussiano
        r : coordenada radial
    """
    m1: np.float64 = 2*np.power(2*a, 0.75) / np.power(np.pi, 0.25)
    m2: np.float64 = np.sqrt(np.power(2,l) / factorial2(2*l+1))
    m3: NDArray = np.power(np.sqrt(2*a)*r, l)
    m4: NDArray = np.exp(-a*np.power(r,2))
    return m1 * m2 * m3 * m4

def GTO(l: int, m: int, alpha: float, r: NDArray, theta: float, phi: float) -> NDArray:
    """ Función de tipo Gaussiana (Gaussian Type Orbital)

    Parámetros
        (l,m) : número cuántico (azimutal, magnético)
        alpha : exponente orbital Gaussiano
        (r,theta,phi) : coordenada (radial, polar, azimutal)
    """
    return R_GTO(l, alpha, r) * sph_harm(l, m, phi, theta)

def R_STO_nG(z: float, d: NDArray, a: NDArray, l: int, r: NDArray) -> NDArray:
    """ Combinación lineal de Gaussianas (parte radial)

    Parámetros
        z : exponente orbital de Slater
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        l : número cuántico azimutal
        r : coordenada radial
    """
    suma = 0
    # calcular cada uno de los k términos de la combinación lineal
    for di, ai in zip(d,a):
        suma += di * R_GTO(l, np.power(z,2)*ai, r) # k-ésimo término
    
    return suma

def STO_nG(z: float, d: NDArray, a: NDArray, l: int, m: int, r: NDArray, theta: float, phi: float) -> NDArray:
    """ Combinación lineal de Gaussianas

    Parámetros
        z : exponente de Slater
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (l,m) : número cuántico (azimutal, magnético)
        (r,theta,phi) : coordenada (radial, polar, azimutal)
    """
    suma = 0
    for di, ai in zip(d,a): # calcular cada uno de los k términos de la suma
        suma += di * GTO(l, m, np.power(z,2)*ai, r, theta, phi) # término k-ésimo
    
    return suma