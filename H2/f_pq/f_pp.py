# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from .t_mn import Tmn # integral cinética
from .v_mn1 import Vmn1 # integral Coulombiana
from .decimal_no_cero import decimal_no_cero
from .valor_truncado import valor_truncado
from .c_pm import cPM # cte. de normalización


def fpp(p: int, d: NDArray, a: NDArray, RA: NDArray, RB: NDArray, ZA: float, ZB: float) -> float:
    """ Elemento de matriz f_pp
    
    Parámetros
        p : elemento de la base, base = {X_1, X_2, X_3, X_4}
            p = 1, 2, 3, 4
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
        (ZA, ZB) : carga del núcleo (A, B)
    """
    TAA: float = Tmn(d, a, RA, RA) # elemento de matriz T_AA y T_BB
    TAB: float = Tmn(d, a, RA, RB) # elemento de matriz T_AB y T_BA

    V1_AA, V1_AB, V1_BB = Vmn1(d, a, RA, RB, RA, ZA) # elementos de matriz V^1_AA, V^1_AB y V^1_BB
    V2_AA, V2_AB, V2_BB = Vmn1(d, a, RA, RB, RB, ZB) # elementos de matriz V^2_AA, V^2_AB y V^2_BB

    # corrección decimal términos Coulombianos
    decimal: int = decimal_no_cero(V1_AB - V2_AB) # primer cifra decimal diferente de cero para V1_AB
    
    if decimal != 0: # solo hay correción si el decimal es diferente de cero
        # matriz V^1
        V1_AA: float = valor_truncado(V1_AA, decimal)
        V1_AB: float = valor_truncado(V1_AB, decimal)
        V1_BB: float = valor_truncado(V1_BB, decimal)
        # matriz V^2
        V2_AA: float = valor_truncado(V2_AA, decimal)
        V2_AB: float = valor_truncado(V2_AB, decimal)
        V2_BB: float = valor_truncado(V2_BB, decimal)

    sgn: int = 1 if p in [1,2] else -1 # signo del coef. de normalización según el elemento de la base
    c2: float = np.square(cPM(d, a, RA, RB, sgn)) # coef. de normalización al cuadrado

    return c2*( 2*(TAA + sgn*TAB) + (V1_AA + V1_BB) + (V2_AA + V2_BB) + 2*sgn*(V1_AB + V2_AB) )