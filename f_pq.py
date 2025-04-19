import numpy as np
from scipy.special import erf
from math import floor, log10

# Referencias.
# [1] A. Szabo y N. S. Ostlund. Modern Quantum Chemistry. Introduction to Advanced Electronic Structure Theory. Dover Publications, 1989.
# [2] T. Helgaker, P. Jørgensen y J. Olsen. Molecular Electronic-Structure Theory. John Wiley & Sons, Ltd, 2000.

# Este trabajo
d = np.array([0.1545314772435472, 0.5354555247125108, 0.4443376735387033]) # coeficientes de contracción
a = np.array([3.423766464751332, 0.6234266839478166, 0.1687830644383135]) # exponentes orbitales Gaussianos

# Szabo & Ostlund
d_Szabo = np.array([0.444635, 0.535328, 0.154329])
a_Szabo = np.array([0.168856, 0.623913, 3.42525])

def GaussNorm(a:int):
    """
        Factor de normalización Gaussiano 1s
    a : exponente Gaussiano
    """
    return np.power(2*a/np.pi, 3/4)

def arg(a:float, b:float, RA:np.ndarray, RB:np.ndarray):
    """
        Argumento del factor pre-exponencial K
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    p = a+b # exponente total
    mu = a*b/p
    RAB2 = np.square(np.linalg.norm(RA-RB))
    return -mu*RAB2

def K(a:float, b:float, RA:np.ndarray, RB:np.ndarray):
    """
        Factor pre-exponencial
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    return np.exp(arg(a, b, RA, RB))

def Spq(a:float, b:float, RA:np.ndarray, RB:np.ndarray):
    """
        Integral de traslape S_pq (normalizada)
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    return GaussNorm(a) * GaussNorm(b) * np.power(np.pi/(a+b), 3/2) * K(a, b, RA, RB)

def Smn(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray):
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

def vdecimal(d:np.ndarray, a:np.ndarray, RA:np.ndarray):
    """ Cifra decimal a la cual se redondearán los elementos de matriz S
    La cifra será del orden de magnitud del primer dígito diferente de cero para la condición de normalización
    """
    SAA = Smn(d, a, RA, RA) # elemento de matriz S_AA y S_BB
    valor = abs(floor(log10(abs(float(SAA) % 1)))+1)
    return valor

def cPM(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray, signo:int):
    """ Constante de normalización c+-
            Psi = c+- (Phi_A +- Phi_B)
            
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB) : coordenada del núcleo (A, B)
    signo : signo de S en la normalización
         1 -> positivo
        -1 -> negativo
    """
    decimal = vdecimal(d, a, RA)
    ord_mag = np.power(10, decimal)
    
    S = np.trunc(Smn(d, a, RA, RB) * ord_mag) / ord_mag # truncar S hasta el primer decimal diferente de cero para S_AA
    return 1/np.sqrt(2*(1 + signo*S))