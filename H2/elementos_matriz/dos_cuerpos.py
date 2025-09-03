# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation

from H2.g_pqrs import gpqrs


def dos_cuerpos(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray, ZA: int, ZB: int) -> list[float]:
    """ Integrales de dos cuerpos para H2 con la base STO-3G

    Parámetros
        d: vector de coeficientes de contracción
        a: vector de exponentes orbitales gaussianos
        (RA, RB): coord. del núcleo (A, B)
        (ZA, ZB): carga del núcleo (A, B)
    """
    g1212: float = gpqrs([1,2,1,2], d, a, RA, RB) # término Coulombiano J11
    g3434: float = gpqrs([3,4,3,4], d, a, RA, RB) # término Coulombiano J22
    g1313: float = gpqrs([1,3,1,3], d, a, RA, RB) # término Coulombiano J12
    g1331: float = gpqrs([1,3,3,1], d, a, RA, RB) # término Coulombiano K12 (intercambio)
    return g1212, g3434, g1313, g1331