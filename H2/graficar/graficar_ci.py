# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from pathlib import Path

import numpy as np
from numpy.typing import NDArray # type annotation
import matplotlib.pyplot as plt


# distancias interatomicas
R_Wolniewicz: NDArray = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 4.0, 4.5, 5.0, 6.0, 8.0])

# Energía (radios de Bohr)
E_Wolniewicz: NDArray = np.array([-0.769635427887, -1.020056664330, -1.124539718008, -1.164935241876, 
                                      -1.174475713565, -1.168583371916, -1.155068736046, -1.138132955488, 
                                      -1.120132112181, -1.102422601703, -1.085791233366, -1.070683223812, 
                                      -1.057326265285, -1.045799653502, -1.016390251364, -1.007993728135, 
                                      -1.003785657939, -1.000835707231, -1.000055604837])


def graficar_CI(R: NDArray, RHF: NDArray, UHF: NDArray, CI: NDArray, EH: NDArray,
                opciones, guardar_img: bool = False, nombre_img: str = 'UHF') -> None:
    """ Graficar E(H2) - 2E(H) para H2:
        - energía RHF, UHF y CI
        - comparación con Wolniewicz

    Parámetros
        R : distancia interatómica
        RHF : energías restringidas ligantes
        UHF : energías no restringidas ligantes
        CI : energías CI ligantes
        EH : energía base del hidrógeno como referencia
        guardar_img : guarda la gráfica como imagen
        nombre_img  : nombre del archivo
    """
        
    # Graficar energía CI

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # 1 fila, 1 columnas, tamaño de figura (15, 5)
    linewidth, fontsize, labelsize, alpha = opciones
    
    ax.axhline(0, color='black', linestyle='dotted') # energía base 2 átomos de hidrógeno
    ax.plot(R, RHF-2*EH, linewidth=linewidth, linestyle='solid', label='RHF', color='#ee2a2a')
    ax.plot(R, UHF-2*EH, linewidth=linewidth, linestyle='dashdot', label='UHF', color='#84ec0f')
    ax.plot(R, CI-2*EH, linewidth=linewidth, linestyle='dashed', label='CI', color='#179eb8')
    ax.plot(R_Wolniewicz, E_Wolniewicz-2*(-0.5), linewidth=3, linestyle='dotted', label='Wolniewicz', color='black', marker='o')
    
    # x config
    xmin, xmax = 0.6, 3.8
    ax.set_xticks(np.arange(0, xmax, 0.5))
    ax.set_xticks(np.arange(0, xmax, 0.25), minor=True)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('Distancia interatómica ($a_0$)', fontsize=fontsize, labelpad=10)
    # y config
    ymin, ymax = -0.215, 0.015
    ax.set_yticks(np.arange(-0.3, ymax, 0.05))
    ax.set_yticks(np.arange(-0.3, ymax, 0.025), minor=True)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(r'E(H$_2$) - 2E(H)', fontsize=fontsize, labelpad=10)
    # plot configplt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,2,3]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=fontsize)
    ax.grid(alpha=alpha, which='both')
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    if guardar_img is True:
        FILE_PATH = Path(__file__).resolve() # ruta de este archivo
        DIR_PATH = FILE_PATH.parents[2] # directorio principal
        DATA_PATH = DIR_PATH / 'imgs' # ruta de las imágenes
        formato: str = 'svg'
        path = str(DATA_PATH / f"{nombre_img}.{formato}")
        
        fig.savefig(path, format=f"{formato}", bbox_inches='tight')

    plt.show()