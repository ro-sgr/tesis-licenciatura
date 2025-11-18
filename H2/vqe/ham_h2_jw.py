# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation
from qiskit.quantum_info import SparsePauliOp

from H2.elementos_matriz import elementos_matriz


def H2_Ham_JW(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray, ZA: float, ZB: float) -> SparsePauliOp:
    """ Hamiltoniano de la molécula de hidrógeno H2 mediante la transformación
        de Jordan-Wigner (JW) dadas las coordenadas y cargas de los núcleos
        para una base STO-kG

    Parámetros
        d: vector de coeficientes de contracción
        a: vector de exponentes orbitales gaussianos
        (RA, RB): coord. del núcleo (A, B)
        (ZA, ZB): carga del núcleo (A, B)
    """

    f11, f33, g1212, g3434, g1313, g1331 = elementos_matriz(d, a, RA, RB, ZA, ZB)
    
    h12: float = (2*f11 + g1212 + 2*g1313 - g1331)/4
    h34: float = (2*f33 + g3434 + 2*g1313 - g1331)/4
    h0: float = f11 + f33 + g1212/4 + g3434/4 + g1313 - g1331/2
    
    hamiltoniano = SparsePauliOp.from_list(
        [
            ("IIII", h0),
            ("IIIZ", -h12), ("IZII", -h12),
            ("IIZI", -h34), ("ZIII", -h34),
            ("IZIZ", g1212/4),
            ("IIZZ", (g1313-g1331)/4), ("ZZII", (g1313-g1331)/4),
            ("ZIIZ", g1313/4), ("XYXY", g1331/4),
            ("XYYX", -g1331/4), ("YXXY", -g1331/4),
            ("YXYX", g1331/4), ("IZZI", g1313/4),
            ("ZIZI", g3434/4),
        ]
    )
    return hamiltoniano