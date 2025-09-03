# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from .s_mn import Smn
from .decimal_no_cero import decimal_no_cero
from .valor_truncado import valor_truncado


def cPM(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray, signo: int) -> float:
    """ Constante de normalización c±
            Psi = c± (Phi_A ± Phi_B)

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
        signo : signo de S en la normalización (1, postivo) y (-1, negativo)
    """
    SAA: float = Smn(d, a, RA, RA) # S_AA y S_BB
    decimal: int = decimal_no_cero(SAA) # primer cifra decimal diferente de cero para S_AA
    SAB: float = Smn(d, a, RA, RB) # S_AB y S_BA
    
    S: float = valor_truncado(SAB, decimal) # S_AB truncado a la precisión decimal de normalización
    
    return 1/np.sqrt(2*(1 + signo*S))