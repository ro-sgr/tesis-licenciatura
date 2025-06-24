# Script para el cálculo de elementos de uno y dos cuerpos
# Molécula de hidrógeno STO-3G

from g_pqrs import *

def elementos_matriz(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray, ZA:int, ZB:int) -> list:
    """ Elementos de matriz para H2 con la base STO-3G

    Parámetros:
        d: vector de coeficientes de contracción
        a: vector de exponentes orbitales gaussianos
        (RA, RB): coord. del núcleo (A, B)
        (ZA, ZB): carga del núcleo (A, B)
    """
    # elementos de un cuerpo
    f11 = fpp(1, d, a, RA, RB, ZA, ZB) # h11 energía cinética (orbital ligante)
    f33 = fpp(3, d, a, RA, RB, ZA, ZB) # h22 energía cinética (orbital antiligante)
    # elementos de dos cuerpos
    g1212 = gpqrs([1,2,1,2], d, a, RA, RB) # término Coulombiano J11
    g3434 = gpqrs([3,4,3,4], d, a, RA, RB) # término Coulombiano J22
    g1313 = gpqrs([1,3,1,3], d, a, RA, RB) # término Coulombiano J12
    g1331 = gpqrs([1,3,3,1], d, a, RA, RB) # término Coulombiano K12 (intercambio)

    return f11, f33, g1212, g3434, g1313, g1331