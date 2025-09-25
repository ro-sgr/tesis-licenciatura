# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

"""
STO-3G (:mod:`H2.vqe`)
================================================================

.. currentmodule:: H2.vqe

Funciones para el Variational Quantum Eigensolver (VQE) para una
molécula de H2 escrita en base STO-3G para diversas separaciones
internucleares
"""

from .ham_h2_jw import H2_Ham_JW
from .funcion_costo import funcion_costo
from .vqe import VQE
from .vqe_h2_jw import VQE_H2_JW

__all__ = [
    "H2_Ham_JW",
    "funcion_costo",
    "VQE",
    "VQE_H2_JW"
]