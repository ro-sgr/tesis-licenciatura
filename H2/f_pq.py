# Script para el cálculo de elementos de un cuerpo
# Molécula de hidrógeno, STO-3G

import numpy as np
from numpy.typing import NDArray # type annotation
from scipy.special import erf

from math import floor, log10

# Referencias.
# [1] A. Szabo y N. S. Ostlund. Modern Quantum Chemistry. Introduction to Advanced Electronic Structure Theory. Dover Publications, 1989.
# [2] T. Helgaker, P. Jørgensen y J. Olsen. Molecular Electronic-Structure Theory. John Wiley & Sons, Ltd, 2000.

#############################################
########## COEFS y EXPONENTES
#############################################

# Szabo & Ostlund
d_Szabo = np.array([0.444635, 0.535328, 0.154329])
a_Szabo = np.array([0.168856, 0.623913, 3.42525])

# Este trabajo
data = np.loadtxt("data/STO3G.csv", delimiter = ",", usecols = (0,1), dtype = ('str'), skiprows = 1) # cargar valores calculados
valores = dict()
for coef in data:
    valores[str(coef[0])] = float(coef[1])
    
d = np.array([valores['d1'], valores['d2'], valores['d3']]) # coeficientes de contracción
a = np.array([valores['a1_2'], valores['a2_2'], valores['a3_2']]) # exponentes orbitales Gaussianos

# distancia interatómica de 1.4 a0 (radios de Bohr)
RA = np.array([0, 0, 0])
RB = np.array([1.4, 0, 0])

# carga nuclear
ZA: float = 1.0
ZB: float = 1.0

#############################################
########## FACTORES
#############################################

def GaussNorm(a: int) -> float:
    """ Factor de normalización Gaussiano 1s

    Parámetro
        a : exponente Gaussiano
    """
    return np.power(2*a/np.pi, 3/4)


def arg(a: float, b: float, RA: NDArray, RB: NDArray) -> float:
    """ Argumento del factor pre-exponencial K
    
    Parámetros
        (a, b)   : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    p: float = a+b # exponente total
    mu: float = a*b/p
    RAB2: float = np.square(np.linalg.norm(RA-RB))
    
    return -mu*RAB2


def K1(a: float, b: float, RA: NDArray, RB: NDArray) -> float:
    """ Factor pre-exponencial (integral 1 cuerpo)
    
    Parámetros
        (a, b)   : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    return np.exp(arg(a, b, RA, RB))

#############################################
########## TRASLAPE
#############################################

def Spq(a: float, b: float, RA: NDArray, RB: NDArray) -> float:
    """ Integral de traslape S_pq (normalizada)
    
    Parámetros
        (a, b)   : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    return GaussNorm(a) * GaussNorm(b) * np.power(np.pi/(a+b), 3/2) * K1(a, b, RA, RB)


def Smn(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray) -> float:
    """ Integral de traslape total S_mn

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
    """
    Mmn: float = 0 # elemento de matriz
    
    for dp, ap in zip(d, a):
        for dq, aq in zip(d, a):
            Mmn += dp * dq * Spq(ap, aq, RA, RB) # elemento de matriz
            
    return Mmn

#############################################
########## CINÉTICA
#############################################

def Tpq(a: NDArray, b: NDArray, RA: NDArray, RB: NDArray) -> float:
    """ Integral cinética T_pq (normalizada)

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
    """
    RAB2: float = np.square(np.linalg.norm(RA-RB)) # cuadrado de diferencia internuclear
    
    return (a*b)/(a+b) * (3 - 2*(a*b)/(a+b)*RAB2 ) * Spq(a, b, RA, RB)


def Tmn(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray) -> float:
    """ Integral cinética total T_mn

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
    """
    Mmn: float = 0.0 # elemento de matriz
    
    for dp, ap in zip(d, a):
        for dq, aq in zip(d, a):
            Mmn += dp * dq * Tpq(ap, aq, RA, RB) # elemento de matriz
            
    return Mmn

#############################################
########## COULOMBIANA
#############################################

def F0(t: float) -> float:
    """ Función de Boys, n=0

    Parámetro
        t : argumento de la función
    """
    return (1/2) * np.sqrt(np.pi/t) * erf(np.sqrt(t))


def RP(a: float, b: float, RA: NDArray, RB: NDArray) -> NDArray:
    """ Coordenada de centro de carga
    
    Parámetro
        (a, b)   : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    p: float = a + b # exponente total
    
    return (a*RA+b*RB)/p


def Vpq_AB(a: float, b: float, RA: NDArray, RB: NDArray, RC: NDArray, ZC: float) -> float:
    """ Integral cinética V_pq(C)

    Parámetros
        (a, b) : exponente orbital Gaussiano
        (RA, RB, RC) : coordenada del núcleo (A, B, C)
        ZC : carga del núcleo C
    """
    factor: float = -2*np.pi/(a+b) * ZC # factor común del término pq
    
    if np.array_equal(RA, RB) and np.array_equal(RB, RC): # todos los núcleos iguales
        Vpq: float = factor
    else: # cualquier otro caso
        RAB2: float = np.square(np.linalg.norm(RA-RB))
        Rp: NDArray = RP(a, b, RA, RB) # coordenada de centro de carga
        RPC2: float = np.square(np.linalg.norm(Rp-RC))
        Vpq: float = factor * K1(a, b, RA, RB) * F0((a+b)*RPC2)
        
    return Vpq


def Vmn1(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray, RC: NDArray, ZC: float) -> list[float]:
    """ Integral cinética V_mn

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB, RC) : coordenada del núcleo (A, B, C)
        ZC : carga del núcleo C
    """
    Mmn1: float = 0.0 # elemento de matriz V^i_AA
    Mmn2: float = 0.0 # elemento de matriz V^i_AB y V^i_BA
    Mmn3: float = 0.0 # elemento de matriz V^i_BB
    
    for dp, ap in zip(d, a):
        for dq, aq in zip(d, a):
            factor: float = dp * dq * GaussNorm(ap) * GaussNorm(aq) # factor común del elemento pq
            Mmn1 += factor * Vpq_AB(ap, aq, RA, RA, RC, ZC)
            Mmn2 += factor * Vpq_AB(ap, aq, RA, RB, RC, ZC)
            Mmn3 += factor * Vpq_AB(ap, aq, RB, RB, RC, ZC)
    
    return Mmn1, Mmn2, Mmn3 # elementos de matriz V^i_AA, V^i_AB, V^i_BB

#############################################
########## Correcciones decimales
#############################################

def decimal_no_cero(valor: float | NDArray) -> int:
    """ Devuelve la posición decimal del primer dígito diferente de cero de un cierto 'valor'
    """
    # if valor == 0:
    #     # valor es cero, i.e. no hay diferencia
    #     decimal: int = 0
    # else:
    #     decimal: int = abs(int(np.floor(np.log10(np.abs(valor)))))

    try:
        decimal: int = abs(floor(log10(abs(valor) % 1)) + 1)
    except ValueError:
        decimal: int = 0
        
    return decimal


def valor_truncado(valor: float | float | NDArray, decimal: int) -> float:
    """ Devuelve un 'valor' truncado hasta una cierta precisión 'decimal'
    """
    ord_mag = np.power(10, decimal) # orden de magnitud
    return np.trunc(valor * ord_mag) / ord_mag

#############################################
########## Otros
#############################################

def cPM(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray, signo: int) -> float:
    """ Constante de normalización c±
            Psi = c± (Phi_A ± Phi_B)

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
        signo : signo de S en la normalización (1, postivo) y (-1, negativo)
    """
    SAA: float = Smn(d, a, RA, RA) # S_AA y S_BB
    decimal: int = decimal_no_cero(SAA) # primer cifra decimal diferente de cero para S_AA
    SAB: float = Smn(d, a, RA, RB) # S_AB y S_BA
    
    S: float = valor_truncado(SAB, decimal) # S_AB truncado a la precisión decimal de normalización
    
    return 1/np.sqrt(2*(1 + signo*S))


def fpp(p: int, d: NDArray, a: NDArray, RA: NDArray, RB: NDArray, ZA: float, ZB: float) -> float:
    """ Elemento de matriz f_pp
    
    Parámetros
        p : elemento de la base, base = {X_1, X_2, X_3, X_4}
            p = 1, 2, 3, 4
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
        (ZA, ZB) : carga del núcleo (A, B)
    """
    TAA: float = Tmn(d, a, RA, RA) # elemento de matriz T_AA y T_BB
    TAB: float = Tmn(d, a, RA, RB) # elemento de matriz T_AB y T_BA

    V1_AA, V1_AB, V1_BB = Vmn1(d, a, RA, RB, RA, ZA) # elementos de matriz V^1_AA, V^1_AB y V^1_BB
    V2_AA, V2_AB, V2_BB = Vmn1(d, a, RA, RB, RB, ZB) # elementos de matriz V^2_AA, V^2_AB y V^2_BB

    # corrección decimal términos Coulombianos
    decimal: int = decimal_no_cero(V1_AB - V2_AB) # primer cifra decimal diferente de cero para V1_AB
    
    if decimal != 0: # solo hay correción si el decimal es diferente de cero
        # matriz V^1
        V1_AA: float = valor_truncado(V1_AA, decimal)
        V1_AB: float = valor_truncado(V1_AB, decimal)
        V1_BB: float = valor_truncado(V1_BB, decimal)
        # matriz V^2
        V2_AA: float = valor_truncado(V2_AA, decimal)
        V2_AB: float = valor_truncado(V2_AB, decimal)
        V2_BB: float = valor_truncado(V2_BB, decimal)

    sgn: int = 1 if p in [1,2] else -1 # signo del coef. de normalización según el elemento de la base
    c2: float = np.square(cPM(d, a, RA, RB, sgn)) # coef. de normalización al cuadrado

    return c2*( 2*(TAA + sgn*TAB) + (V1_AA + V1_BB) + (V2_AA + V2_BB) + 2*sgn*(V1_AB + V2_AB) )
