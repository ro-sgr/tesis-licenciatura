# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation


def RP(a: float, b: float, RA: NDArray, RB: NDArray) -> NDArray:
    """ Coordenada de centro de carga
    
    Parámetro
        (a, b)   : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    p: float = a + b # exponente total
    
    return (a*RA+b*RB)/p