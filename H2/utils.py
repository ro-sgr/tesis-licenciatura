from pathlib import Path

import numpy as np
from numpy.typing import NDArray # type annotation


def guardar(nombre: str, data: list | dict, encabezado: str = None) -> None:
    """ Guardar lista de datos

    Parámetros
        nombre : nombre del archivo
        encabezado : nombre de la columna
        data   : lista / diccionario de datos por ser guardados
    """
    FILE_PATH = Path(__file__).resolve() # ruta de este archivo
    DIR_PATH = FILE_PATH.parents[1] # directorio principal
    DATA_PATH = DIR_PATH / 'data' # ruta de los datos
    path = str(DATA_PATH / f"{nombre}")

    with open(path, "w") as file:
        # datos son una lista
        if type(data) is list:
            if encabezado is None:
                encabezado = nombre
            file.write(f"{encabezado}\n")
            for item in data:
                file.write(f"{item}\n")
        # datos son un diccionario
        if type(data) is dict:
            file.write("nombre, valor\n")
            for k, v in data.items():
                file.write(f"{k}, {v}\n")



def cargar(nombre: str,
           delimiter: str = ",",
           dtype: tuple[str] = ('float'),
           skiprows: int = 1,
           usecols: int | tuple[int] = None,
           unpack: bool = True) -> NDArray[float]:
    """ Cargar conjunto de datos

    Parámetro
        nombre : nombre del archivo
    """
    FILE_PATH = Path(__file__).resolve() # ruta de este archivo
    DIR_PATH = FILE_PATH.parents[1] # directorio principal
    DATA_PATH = DIR_PATH / 'data' # ruta de los datos
    
    return np.loadtxt(DATA_PATH / f"{nombre}",
                      delimiter = delimiter,
                      dtype = dtype,
                      skiprows = skiprows,
                      usecols = usecols,
                      unpack = unpack)
	