# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def compararSTO(nombre: str, formato: str, ref: dict, datasets: list, xticks: list, yticks: list) -> None:
    """ Gráfica de una función principal f(x) y múltiples funciones gi(x)

    nombre : nombre del archivo .formato (si 'None' no se guarda la gráfica)
    ref : conjunto de datos de referencia
        ref.keys() = ['x', 'y', 'label', 'color']
    datasets : lista de conjuntos por comparar [data1, data2, ...]
        dataN.keys() = ['y', 'label', 'linestyle', 'linewidth', 'marker', 'markersize', 'color', 'N']
            N : [a,b,...]
                1er conjunto -> se consideran únicamente cada 'a' número de elementos
                2do conjunto -> se consideran únicamente cada 'b' número de elementos
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 fila, 2 columnas, tamaño de figura (15, 5)

    # Comparación función de Slater con Gaussianas
    axs[0].plot(ref['x'], ref['y'], label=ref['label'], color=ref['color']) # función principal
    for data in datasets: # funciones aproximadas
        x = ref['x'][::data['N']]
        y = data['y'][::data['N']]
        axs[0].plot(x, y, label=data['label'], linestyle=data['linestyle'], linewidth=data['linewidth'],
                    marker=data['marker'], markersize=data['markersize'], color=data['color'])
    axs[0].set_ylabel(r'$R_{1s}$', fontsize='xx-large', labelpad=10)
    axs[0].set_xticks(xticks[0])
    axs[0].set_yticks(yticks[0])

    # Comparación de las funciones de distribución radiales de Slater con Gaussianas
    k = 4 * np.pi * np.power(ref['x'], 2)
    axs[1].plot(ref['x'], k * np.power(ref['y'], 2), label='Slater', color='k') # función principal
    for data in datasets: # funciones aproximadas
        x = ref['x'][::data['N']]
        k = 4 * np.pi * np.power(x, 2)
        y = data['y'][::data['N']]
        axs[1].plot(x, k * np.power(y, 2), label=data['label'], linestyle=data['linestyle'], linewidth=data['linewidth'],
                    marker=data['marker'], markersize=data['markersize'], color=data['color'])
    axs[1].set_ylabel(r'$4\pi r^2 |R_{1s}|^2$', fontsize='xx-large', labelpad=10)
    axs[1].set_xticks(xticks[1])
    axs[1].set_yticks(yticks[1])

    for ax in axs:
        ax.set_xlabel('Radio ($a_0$)', fontsize='xx-large', labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize='xx-large')
        ax.legend(fontsize=15, loc='upper right')
        ax.grid(alpha=alpha, which='both')

    fig.subplots_adjust(wspace=0.25)
    
    if nombre is not None: # guardar
        FILE_PATH = Path(__file__).resolve() # ruta de este archivo
        DIR_PATH = FILE_PATH.parents[2] # directorio principal
        DATA_PATH = DIR_PATH / 'imgs' # ruta de las imágenes
        path = str(DATA_PATH / f"{nombre}.{formato}")
        
        fig.savefig(path, format=f"{formato}", bbox_inches='tight')
        
    plt.show()