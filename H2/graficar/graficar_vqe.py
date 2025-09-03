# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def graficarVQE(nombre: str, ref: dict, datasets: list):
    """ Gráfica de datos obtenidos para el VQE (unidades atómicas y Angstroms)

    nombre : nombre del archivo .svg (si 'None' no se guarda la gráfica)
    ref : diccionario con datos de referencia (e.g. Kandala)
        x : distancias en Angstroms
        y : energía en Hartrees
    datasets -> [data1, data2, ... ] : lista de diccionarios con cada conjunto de datos por graficar
        x : distancias en unidades atómicas de distancia
        y : energía en Hartrees
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    factor = 0.529177249 # unidades atómicas de distancia a Angstroms
    
    # Unidades atómicas
    axs[0].plot(ref['x']/factor, ref['y'], label=ref['label'], linewidth=ref['linewidth'], linestyle=ref['linestyle'], marker=ref['marker'], color=ref['color'])
    for data in datasets:
        axs[0].plot(data['x'], data['y'], label=data['label'], linewidth=data['linewidth'], linestyle=data['linestyle'], marker=data['marker'], color=data['color'])
    axs[0].set_xlabel("Distancia interatómica (Unidades Atómicas)", fontsize='xx-large', labelpad=10)
    axs[0].set_ylabel("Energía (Hartrees)", fontsize='xx-large', labelpad=10)

    # Angstroms
    axs[1].plot(ref['x'], ref['y'], label=ref['label'], linewidth=ref['linewidth'], linestyle=ref['linestyle'], marker=ref['marker'], color=ref['color'])
    for data in datasets:
        axs[1].plot(data['x']*factor, data['y'], label=data['label'], linewidth=data['linewidth'], linestyle=data['linestyle'], marker=data['marker'], color=data['color'])
    axs[1].set_xlabel("Distancia interatómica (Angstrom)", fontsize='xx-large', labelpad=10)
    
    orden = np.arange(0,len(datasets)+1) # añade leyendas en el orden que se insertaron los conjuntos de datos
    for ax in axs:
        handles, labels=ax.get_legend_handles_labels()
        ax.legend([handles[idx] for idx in orden], [labels[idx] for idx in orden], fontsize='xx-large', loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize='xx-large')
        ax.grid()
    
    fig.subplots_adjust(wspace=0.2) # espacio entre figuras
    
    if nombre is not None: # guardar
        FILE_PATH = Path(__file__).resolve() # ruta de este archivo
        DIR_PATH = FILE_PATH.parents[2] # directorio principal
        DATA_PATH = DIR_PATH / 'imgs' # ruta de las imágenes
        formato: str = 'svg'
        path = str(DATA_PATH / f"{nombre}.{formato}")
        
        fig.savefig(path, format=f"{formato}", bbox_inches='tight')
        
    plt.show()