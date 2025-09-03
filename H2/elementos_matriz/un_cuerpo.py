# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation

from H2.f_pq import fpp


def un_cuerpo(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray, ZA: int, ZB: int) -> list[float]:
    """ Integrales de un solo cuerpo para H2 con la base STO-3G

    Parámetros
        d: vector de coeficientes de contracción
        a: vector de exponentes orbitales gaussianos
        (RA, RB): coord. del núcleo (A, B)
        (ZA, ZB): carga del núcleo (A, B)
    """
    f11: float = fpp(1, d, a, RA, RB, ZA, ZB) # h11 energía cinética (orbital ligante)
    f33: float = fpp(3, d, a, RA, RB, ZA, ZB) # h22 energía cinética (orbital antiligante)
    return f11, f33