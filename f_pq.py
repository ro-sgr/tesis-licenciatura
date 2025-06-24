# Script para el cálculo de elementos de un cuerpo
# Molécula de hidrógeno STO-3G

import numpy as np
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
data = np.loadtxt("data/STO3G.csv", delimiter=",", usecols=(0,1), dtype=('str')) # cargar valores calculados
valores = dict()
for valor in data:
    valores[str(valor[0])] = float(valor[1])
    
d = np.array([valores['d1'], valores['d2'], valores['d3']]) # coeficientes de contracción
a = np.array([valores['a1_2'], valores['a2_2'], valores['a3_2']]) # exponentes orbitales Gaussianos

# distancia interatómica de 1.4 a0 (radios de Bohr)
RA = np.array([0, 0, 0])
RB = np.array([1.4, 0, 0])

# carga nuclear
ZA, ZB = 1, 1

#############################################
########## FACTORES
#############################################

def GaussNorm(a:int) -> np.float64:
    """ Factor de normalización Gaussiano 1s

    Parámetro
        a : exponente Gaussiano
    """
    return np.power(2*a/np.pi, 3/4)

def arg(a:float, b:float, RA:np.ndarray, RB:np.ndarray) -> np.float64:
    """ Argumento del factor pre-exponencial K
    
    Parámetros
        (a, b) : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    p = a+b # exponente total
    mu = a*b/p
    RAB2 = np.square(np.linalg.norm(RA-RB))
    return -mu*RAB2

def K1(a:float, b:float, RA:np.ndarray, RB:np.ndarray) -> np.float64:
    """ Factor pre-exponencial (integral 1 cuerpo)
    
    Parámetros
        (a, b) : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    return np.exp(arg(a, b, RA, RB))

#############################################
########## TRASLAPE
#############################################

def Spq(a:float, b:float, RA:np.ndarray, RB:np.ndarray) -> np.float64:
    """ Integral de traslape S_pq (normalizada)
    
    Parámetros
        (a, b) : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    return GaussNorm(a) * GaussNorm(b) * np.power(np.pi/(a+b), 3/2) * K1(a, b, RA, RB)

def Smn(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray) -> np.float64:
    """ Integral de traslape total S_mn

    Parámetros
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

#############################################
########## CINÉTICA
#############################################

def Tpq(a:np.ndarray, b:np.ndarray, RA:np.ndarray, RB:np.ndarray) -> np.float64:
    """ Integral cinética T_pq (normalizada)

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
    """
    RAB2 = np.square(np.linalg.norm(RA-RB)) # cuadrado de diferencia internuclear
    return (a*b)/(a+b) * (3 - 2*(a*b)/(a+b)*RAB2 ) * Spq(a, b, RA, RB)

def Tmn(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray) -> np.float64:
    """ Integral cinética total T_mn

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
    """
    Mmn = 0 # elemento de matriz
    k = len(d) # tamaño de la k-contracción
    for p in range(k):
        for q in range(k):
            Mmn += d[p] * d[q] * Tpq(a[p], a[q], RA, RB) # elemento de matriz
    return Mmn

#############################################
########## COULOMBIANA
#############################################

def F0(t:float) -> np.float64:
    """ Función de Boys, n=0

    Parámetro
        t : argumento de la función
    """
    return (1/2) * np.sqrt(np.pi/t) * erf(np.sqrt(t))

def RP(a:float, b:float, RA:np.ndarray, RB:np.ndarray) -> np.ndarray:
    """ Coordenada de centro de carga
    
    Parámetro
        (a, b) : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    p = a + b # exponente total
    return (a*RA+b*RB)/p

def Vpq_AB(a:float, b:float, RA:np.ndarray, RB:np.ndarray, RC:np.ndarray, ZC:float) -> np.float64:
    """ Integral cinética V_pq(C)

    Parámetros
        (a, b) : exponente orbital Gaussiano
        (RA, RB, RC) : coordenada del núcleo (A, B, C)
        ZC : carga del núcleo C
    """
    factor = -2*np.pi/(a+b) * ZC # factor común del término pq
    
    if np.array_equal(RA, RB) and np.array_equal(RB, RC): # todos los núcleos iguales
        Vpq = factor
    else: # cualquier otro caso
        RAB2 = np.square(np.linalg.norm(RA-RB))
        Rp = RP(a, b, RA, RB) # coordenada de centro de carga
        RPC2 = np.square(np.linalg.norm(Rp-RC))
        Vpq = factor * K1(a, b, RA, RB) * F0((a+b)*RPC2)
    return Vpq

def Vmn1(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray, RC:np.ndarray, ZC:float) -> np.float64:
    """ Integral cinética V_mn

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB, RC) : coordenada del núcleo (A, B, C)
        ZC : carga del núcleo C
    """
    Mmn1 = 0 # elemento de matriz V^i_AA
    Mmn2 = 0 # elemento de matriz V^i_AB y V^i_BA
    Mmn3 = 0 # elemento de matriz V^i_BB
    k = len(d)

    for p in range(k):
        for q in range(k):
            factor = d[p] * d[q] * GaussNorm(a[p]) * GaussNorm(a[q]) # factor común del término p, q
            Mmn1 += factor * Vpq_AB(a[p], a[q], RA, RA, RC, ZC)
            Mmn2 += factor * Vpq_AB(a[p], a[q], RA, RB, RC, ZC)
            Mmn3 += factor * Vpq_AB(a[p], a[q], RB, RB, RC, ZC)
    
    return [Mmn1, Mmn2, Mmn3] # elementos de matriz (V^i_AA, V^i_AB, V^i_BB)


#############################################
########## Correcciones decimales
#############################################

def SAB_vdecimal(d:np.ndarray, a:np.ndarray, RC:np.ndarray) -> int:
    """ Precisión decimal integral de traslape
    Determina la cifra decimal a la cual se redondearán los elementos de matriz S
    La cifra será del orden de magnitud del primer dígito diferente de cero para la condición de normalización

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        RC : coordenada del núcleo C
    """
    SCC = Smn(d, a, RC, RC) # elemento de matriz S_AA y S_BB
    valor = abs(floor(log10(abs(float(SCC) % 1)))+1)
    return valor

def VAB_vdecimal(VA_AB:np.float64, VB_AB:np.float64) -> int:
    """ Precisión decimal integral Coulombiana
    Determina la cifra decimal a la cual se redondearán los elementos de la antidiagonal de las matrices V^A y V^B
    La cifra será del orden de magnitud del primer dígito diferente de cero para la condición de normalización

    Parámetros
        Vi_AB : elemento de matriz V^i_AB
    """
    str_a, str_b = str(VA_AB), str(VB_AB) # convertir a texto
    str_a, str_b = str_a.split('.',1)[1], str_b.split('.',1)[1] # separar en parte entera y decimal (se conserva la decimal)
    len_a, len_b = len(str_a), len(str_b) # largo de cada valor
    min_len = min(len_a, len_b) # largo más pequeño para evitar IndexError

    for i in range(min_len):
        if str_a[i] != str_b[i]:
            return i # valor decimal donde difieren

    # dígitos coinciden hasta el decimal 'min_len', revisa si largos son diferentes
    if len_a != len_b:
        return min_len

    return -1 # valores son idénticos

#############################################
########## Otros
#############################################

def cPM(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray, signo:int) -> np.float64:
    """ Constante de normalización c±
            Psi = c± (Phi_A ± Phi_B)

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
        signo : signo de S en la normalización (1, postivo) y (-1, negativo)
    """
    SAB_decimal = SAB_vdecimal(d, a, RA) # cifra decimal de error para S_AB
    SAB_ord_mag = np.power(10, SAB_decimal) # orden de magnitud
    
    S = np.trunc(Smn(d, a, RA, RB) * SAB_ord_mag) / SAB_ord_mag # truncar S hasta el primer decimal diferente de cero para S_AA
    return 1/np.sqrt(2*(1 + signo*S))

def fpp(p:int, d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray, ZA:float, ZB:float) -> np.float64:
    """ Elemento de matriz f_pp
    
    Parámetros
        p : elemento de la base, base = {X_1, X_2, X_3, X_4}
            p = 1, 2, 3, 4
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
        (ZA, ZB) : carga del núcleo (A, B)
    """
    TAA = Tmn(d, a, RA, RA) # elemento de matriz T_AA y T_BB
    TAB = Tmn(d, a, RA, RB) # elemento de matriz T_AB y T_BA

    V1_AA, V1_AB, V1_BB = Vmn1(d, a, RA, RB, RA, ZA) # elementos de matriz V^1_AA, V^1_AB y V^1_BB
    V2_AA, V2_AB, V2_BB = Vmn1(d, a, RA, RB, RB, ZB) # elementos de matriz V^2_AA, V^2_AB y V^2_BB

    # corrección decimal términos Coulombianos
    VAB_decimal = VAB_vdecimal(V1_AB, V2_AB) # cifra decimal de error para V^i_AB
    V1_AB = V1_AB if (VAB_decimal == -1) else np.round(V1_AB, VAB_decimal) # si -1 no hay corrección
    V2_AB = V2_AB if (VAB_decimal == -1) else np.round(V2_AB, VAB_decimal) # si -1 no hay corrección

    sgn = 1 if p in [1,2] else -1 # signo del coef. de normalización según el elemento de la base
    c2 = np.square(cPM(d, a, RA, RB, sgn)) # coef. de normalización al cuadrado
    term = c2*( 2*(TAA + sgn*TAB) + (V1_AA + V1_BB) + (V2_AA + V2_BB) + 2*sgn*(V1_AB + V2_AB) )
    return term