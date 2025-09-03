# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

"""
STO-3G (:mod:`H2.graficar`)
=====================================================================

.. currentmodule:: H2.graficar

Funciones para graficar funciones de tipo Slater, elementos de matriz,
energías RHF y el Vatiational Quantum Eigensolver
"""

from .comparar_sto import compararSTO
from .graficar_elementos_matriz import graficar_elementos_matriz
from .graficar_rhf import graficar_RHF
from .graficar_vqe import graficar_VQE

__all__ = [
    "compararSTO",
    "graficar_elementos_matriz",
    "graficar_RHF",
    "graficar_VQE"
]