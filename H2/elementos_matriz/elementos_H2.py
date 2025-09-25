# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation

# Coeficientes STO3G
from H2.sto_kG import d # coeficientes de contracción
from H2.sto_kG import a2 as a # exponentes orbitales Gaussianos

from H2.utils import guardar, cargar
from .elementos_matriz_rango import elementos_matriz_rango


def elementos_H2(
        modo: str = 'cargar',
        sufijo: str = 'H2_STO3G',
        intervalo: list = None,
        coefs: list[list] = [d,a],
        carga_nuc: list[float] = [1, 1] # Z_A, Z_B
    ) -> list[float]:
    """ Cargar o calcular elementos de matriz para H2 en base STO-kG dado un intervalo de distancias internucleares
    
    Formato: CSV

    Parámetros
        modo:
            'cargar' --> emplear los elementos de matriz de los archivos del directorio 'data'
            'calcular_rango' --> calcular los elementos de matriz en un intervalo de distancias internucleares
            'calcular_unica' --> calcular los elementos de matriz para una única distancia interatómica
        sufijo: nombre base de los archivos por cargar / guardar, e.g. '{sufijo}_nombre.csv'
        intervalo: [inicio, fin, paso]
            Indica la distancia internuclear de inicio, fin y paso a considerar para calcular los elementos de matriz
            Los elementos de matriz se guardan en el directorio 'data' como '{sufijo}_nombre.csv'
            Por defecto la separación es R=1.4a0
        coefs: [[d1, d2, ..., dk], [a1, a2, ..., ak]]
            Conjunto de coeficientes para una k-contracción de funciones Gaussianas
            Por defecto los coeficientes para STO-3G
        carga_nuc: [ZA, ZB]
            Carga de los núcleos A y B. Por defecto [1,1]
    """
    
    # cargar elementos de matriz h11, h22, J11, J22, J12, K12 y hnuc
    if modo == 'cargar':
        distancias: NDArray = cargar('H2_distancias.csv', delimiter = ",", dtype = ('float'), skiprows = 1)
        h11: NDArray = cargar(f"{sufijo}_h11.csv", delimiter = ",", dtype = ('float'), skiprows = 1)
        h22: NDArray = cargar(f"{sufijo}_h22.csv", delimiter = ",", dtype = ('float'), skiprows = 1)
        J11: NDArray = cargar(f"{sufijo}_J11.csv", delimiter = ",", dtype = ('float'), skiprows = 1)
        J22: NDArray = cargar(f"{sufijo}_J22.csv", delimiter = ",", dtype = ('float'), skiprows = 1)
        J12: NDArray = cargar(f"{sufijo}_J12.csv", delimiter = ",", dtype = ('float'), skiprows = 1)
        K12: NDArray = cargar(f"{sufijo}_K12.csv", delimiter = ",", dtype = ('float'), skiprows = 1)
        hnuc: NDArray = cargar(f"{sufijo}_hnuc.csv", delimiter = ",", dtype = ('float'), skiprows = 1)
        print("Elementos de matriz y distancias interatómicas cargadas.")
        
    # calcular elementos de matriz h11, h22, J11, J22, J12, K12 y hnuc, dado un intervalo de distancias interatómicas
    elif modo == 'calcular_rango':
        distancias, h11, h22, J11, J22, J12, K12, hnuc = elementos_matriz_rango(coefs, carga_nuc, intervalo)
        print('Elementos de matriz calculados para el intervalo de distancias.')
        # guardar
        guardar(f"{sufijo}_distancias.csv", distancias)
        guardar(f"{sufijo}_h11.csv", h11)
        guardar(f"{sufijo}_h22.csv", h22)
        guardar(f"{sufijo}_J11.csv", J11)
        guardar(f"{sufijo}_J22.csv", J22)
        guardar(f"{sufijo}_J12.csv", J12)
        guardar(f"{sufijo}_K12.csv", K12)
        guardar(f"{sufijo}_hnuc.csv", hnuc)
        print(f"Elementos de matriz y distancias interatómicas guardadas en el directorio 'data' bajo el sufijo {sufijo}.")

    else:
        print("Las únicas opciones posibles son:")
        print(" cargar : emplear los elementos de matriz de los archivos del directorio 'data'")
        print(" calcular_rango : calcular los elementos de matriz en un intervalo de distancias internucleares")
        return None
        
    return distancias, h11, h22, J11, J22, J12, K12, hnuc