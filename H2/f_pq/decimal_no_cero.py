# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from math import floor, log10

import numpy as np
from numpy.typing import NDArray # type annotation


def decimal_no_cero(valor: float | NDArray) -> int:
    """ Devuelve la posición decimal del primer dígito diferente de cero de un cierto 'valor'
    """
    # if valor == 0:
    #     # valor es cero, i.e. no hay diferencia
    #     decimal: int = 0
    # else:
    #     decimal: int = abs(int(np.floor(np.log10(np.abs(valor)))))

    try:
        decimal: int = abs(floor(log10(abs(valor) % 1)) + 1)
    except ValueError:
        decimal: int = 0
        
    return decimal