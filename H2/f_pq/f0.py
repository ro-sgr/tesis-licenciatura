# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from scipy.special import erf


def F0(t: float) -> float:
    """ Función de Boys, n=0

    Parámetro
        t : argumento de la función
    """
    return (1/2) * np.sqrt(np.pi/t) * erf(np.sqrt(t))