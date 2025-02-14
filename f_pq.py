import numpy as np
from scipy.special import erf

def GaussNorm(a : int):
    """
        Factor de normalización Gaussiano 1s
    a : exponente Gaussiano
    """
    return np.power(2*a/np.pi, 3/4)

def arg(a : float, b : float, RA : np.array, RB : np.array):
    """
        Argumento del factor pre-exponencial K
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    p = a+b # exponente total
    mu = a*b/p
    RAB2 = np.square(np.linalg.norm(RA-RB))
    return -mu*RAB2

def K(a : float, b : float, RA : np.array, RB : np.array):
    """
        Factor pre-exponencial
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    return np.exp(arg(a, b, RA, RB))

def Spq(a : float, b : float, RA : np.array, RB : np.array):
    """
        Integral de traslape S_pq (normalizada)
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    return GaussNorm(a) * GaussNorm(b) * np.power(np.pi/(a+b), 3/2) * K(a, b, RA, RB)

def Smn(d : np.array, a : np.array, RA : np.array, RB : np.array):
    """
        Integral de traslape total S_mn
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB) : coordenada del núcleo (A, B)
    """
    Mmn = 0 # elemento de matriz
    k = len(d)
    for p in range(k):
        for q in range(k):
            Mmn += d[p] * d[q] * Spq(a[p], a[q], RA, RB) # elemento de matriz
    return Mmn