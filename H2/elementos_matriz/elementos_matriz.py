# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation

from .un_cuerpo import un_cuerpo
from .dos_cuerpos import dos_cuerpos


def elementos_matriz(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray, ZA: int, ZB: int) -> list[float]:
    """ Elementos de matriz para H2 con la base STO-3G

    Parámetros
        d: vector de coeficientes de contracción
        a: vector de exponentes orbitales gaussianos
        (RA, RB): coord. del núcleo (A, B)
        (ZA, ZB): carga del núcleo (A, B)
    """
    # elementos de un cuerpo
    f11, f33 = un_cuerpo(d, a, RA, RB, ZA, ZB)
    # elementos de dos cuerpos
    g1212, g3434, g1313, g1331 = dos_cuerpos(d, a, RA, RB, ZA, ZB)
    
    return f11, f33, g1212, g3434, g1313, g1331