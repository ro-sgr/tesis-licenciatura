# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation

from H2.f_pq import GaussNorm
from .ac_bd import ACBD


def V12(a: float, b: float, c: float, d: float, RA: NDArray, RB: NDArray, RC: NDArray, RD: NDArray) -> float:
    """ Integral de dos electrones total (normalizada)
        
    (a, b, c, d) : exponentes orbitales Gaussianos
    (RA, RB, RC, RD) : coordenadas del núcleo (A, B, C, D)
        A, C : asociados al electrón 1
        B, D : asociados al electrón 2
    """
    return GaussNorm(a) * GaussNorm(b) * GaussNorm(c) * GaussNorm(d) * ACBD(a, b, c, d, RA, RB, RC, RD)