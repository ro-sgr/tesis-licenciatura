# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

"""
========================================
STO-kG molécula de hidrógeno (:mod:`H2`)
========================================

H2 es una paquetería desarrollada para calcular los coeficientes para la función de
onda de la molécula de hidrógeno compuesta de k-contracciones de Gaussianas (STO-kG).

Esta paquetería también permite el cálculo de las energías restringidas y no restringidas
mediante el método de Hartree Fock (RHF & UHF).

Para una explicación detallada del origen de las funciones, refiérase a los cuadernos de
Jupyter (archivos con extensión `.ipynb`) en donde se da una explicación por completo de
cada parte del código.

Considere a la paquetería como autocontenida, i.e. su propósito es el de funcionar con el
presente código, por lo que no ha sido optimizada de forma alguna pensando en aplicaciones
ajenas a este trabajo.

.. currentmodule:: H2

Submodules
----------

.. autosummary::
    :toctree:

    sto_kG
    f_pq
    g_pqrs
    elementos_matriz
    graficar
    utils
"""