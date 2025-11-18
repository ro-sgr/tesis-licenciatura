# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from pathlib import Path

import numpy as np
from numpy.typing import NDArray # type annotation
import matplotlib.pyplot as plt


def graficar_RHF(R: NDArray, y: list[NDArray], EH: NDArray, opciones, guardar_img: bool = False, nombre: str = 'RHF') -> None:
    """ Graficar energía RHF para H2

    Parámetros
        R : distancia interatómica
        y : [RHF1, RHF2]
        EH : energía base del hidrógeno como referencia
        guardar_img : guarda la gráfica como imagen
    """

    RHF1, RHF2 = y
    linewidth, fontsize, labelsize, alpha = opciones
    
    #####################
    ###### Gráfica ######
    #####################
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))  # 1 fila, 1 columnas, tamaño de figura (15, 5)
    linewidth, fontsize, labelsize, alpha=5, 22, 18, 0.4
    
    ax.plot(R, RHF1, label='RHF(+)', linestyle='solid', linewidth=linewidth, color='#21b0fe') # energías RHF orbital ligante
    ax.plot(R[3:], RHF2[3:], label='RHF(-)', linestyle='dashed', linewidth=linewidth, color='#fed700') # energías RHF orbital antiligante
    ax.axhline(y=2*EH, color='black', linestyle='dotted') # energía base 2 átomos de hidrógeno (STO-3G)
    
    # x config
    ax.set_xlim(-0.05, 7.5)
    ax.set_xlabel(r'Distancia interatómica ($a_0$)', fontsize=fontsize)
    ax.set_xticks(np.arange(0,8,1), minor=True)
    # y config
    ax.set_ylim(-1.19, 0.05)
    ax.set_ylabel(r'Energía (E$_\mathrm{h}$)', fontsize=fontsize, labelpad=10)
    ax.set_yticks(np.arange(-1.1, 0.05, 0.2))
    # plot config
    ax.legend(fontsize='xx-large')
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.grid(alpha=alpha, which='both')

    if guardar_img is True:
        FILE_PATH = Path(__file__).resolve() # ruta de este archivo
        DIR_PATH = FILE_PATH.parents[2] # directorio principal
        DATA_PATH = DIR_PATH / 'imgs' # ruta de las imágenes
        formato: str = 'svg'
        path = str(DATA_PATH / f"{nombre}.{formato}")
        
        fig.savefig(path, format=f"{formato}", bbox_inches='tight')

    plt.show()