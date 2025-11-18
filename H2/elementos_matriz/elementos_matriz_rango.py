# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation

# Coeficientes STO3G
from H2.sto_kG import d # coeficientes de contracción
from H2.sto_kG import a2 as a # exponentes orbitales Gaussianos

from .elementos_matriz import elementos_matriz


def elementos_matriz_rango(intervalo: list, coefs: list = [d,a], carga_nuc: list = [1,1]) -> tuple[list[float]]:
    """ Lista de elementos de matriz de H2 con la base STO-3G para diversas distancias
    internucleares en el intervalo (inicio, fin) con incrementos de 'paso' 

    Parámetros
        intervalo: [inicio, fin, paso]
            Indica la distancia internuclear de inicio, fin y paso a considerar para calcular los elementos de matriz
        coefs: [[d1, d2, ..., dk], [a1, a2, ..., ak]]
            Conjunto de coeficientes para una k-contracción de funciones Gaussianas
            Por defecto los coeficientes para STO-3G
        carga_nuc: [ZA, ZB]
            Carga de los núcleos A y B. Por defecto [1,1]
    """
    h11, h22, J11, J22, J12, K12, hnuc = [], [], [], [], [], [], []

    inicio, fin, paso = intervalo
    distancias: NDArray = np.arange(inicio, fin, paso) # distancias internucleares
    decimal: int = len(str(paso).split('.')[1]) # número de decimales en el paso
    distancias: NDArray = np.round(distancias, decimal) # remover error numérico redondeando al número de decimales previsto
    # En esto último se asume que 'inicio', 'fin' y 'paso' tienen a lo más el mismo número de cifras decimales
    
    for x in distancias:
        RA: NDArray = np.array([0, 0, 0]) # núcleo A fijo en 0
        RB: NDArray = np.array([x, 0, 0]) # posición del núcleo B
        h11_val, h22_val, J11_val, J22_val, J12_val, K12_val = elementos_matriz(d, a, RA, RB, ZA, ZB)
        hnuc_val: float = 1/(np.linalg.norm(RA-RB)) # repulsión nuclear
        h11.append(h11_val)
        h22.append(h22_val)
        J11.append(J11_val)
        J22.append(J22_val)
        J12.append(J12_val)
        K12.append(K12_val)
        hnuc.append(hnuc_val)
    
    # convertir en arreglos de numpy
    h11, h22, J11, J22, J12, K12, hnuc = [np.array(h11), np.array(h22), np.array(J11), np.array(J22), np.array(J12), np.array(K12), np.array(hnuc)]

    return distancias, h11, h22, J11, J22, J12, K12, hnuc