# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation

from .v_12 import V12


def Vmn2(d : NDArray, a: NDArray, RA: NDArray, RB: NDArray, RC: NDArray, RD: NDArray) -> float:
    """ Elemento de matriz de interacción de dos electrones

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB, RC, RD) : coordenadas del núcleo (A, B, C, D)
    """
    Mijkl: float = 0.0 # elemento de tensor
    
    L: int = len(d)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for l in range(L):
                    Mijkl += d[i] * d[j] * d[k] * d[l] * V12(a[i], a[j], a[k], a[l], RA, RB, RC, RD)
                    
    return Mijkl