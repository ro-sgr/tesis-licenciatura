# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation
from scipy.special import sph_harm_y as sph_harm # armónicos esféricos

from .r_gto import R_GTO


def χ_GTO(l: int, m: int, α: float, r: NDArray, θ: NDArray, ϕ: NDArray) -> NDArray:
    """ Función de tipo Gaussiana (Gaussian Type Orbital)

    Parámetros
        (l, m) : número cuántico (azimutal, magnético)
        α : exponente orbital Gaussiano
        (r, θ, ϕ) : coordenada (radial, polar, azimutal)
    """
    return R_GTO(l, α, r) * sph_harm(l, m, ϕ, θ)