# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation

from .gauss_norm import GaussNorm
from .v_pq_AB import Vpq_AB


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