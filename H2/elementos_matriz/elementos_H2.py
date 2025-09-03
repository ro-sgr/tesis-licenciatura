# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from numpy.typing import NDArray # type annotation

from H2.utils import guardar, cargar
from .elementos_matriz_distancias import elementos_matriz_distancias


def elementos_H2(modo: str = 'cargar', sufijo: str = 'H2_STO3G', intervalo: list = None) -> list[float]:
    """ Cargar o calcular elementos de matriz para H2 dado un cierto intervalo

    Parámetros
        modo:
            'cargar' -> emplear los elementos de matriz de los archivos del directorio 'data'
            'calcular' -> calcular los elementos de matriz
        sufijo: nombre base de los archivos por cargar / guardar
        intervalo: [inicio, fin, paso]
            indica la distancia interatómica de inicio, fin y paso a considerar para calcular los elementos de matriz
            los elementos de matriz se guardan como .csv en el directorio 'data' bajo el nombre 'H2_nombre'
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
        
    # calcular elementos de matriz h11, h22, J11, J22, J12, K12 y hnuc
    elif modo == 'calcular':
        inicio, fin, paso = intervalo
        distancias, h11, h22, J11, J22, J12, K12, hnuc = elementos_matriz_distancias(inicio, fin, paso, d, a, ZA, ZB)
        print('Elementos de matriz calculados.')
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
        
    return distancias, h11, h22, J11, J22, J12, K12, hnuc