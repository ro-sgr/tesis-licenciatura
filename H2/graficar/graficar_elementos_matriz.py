# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def graficar_elementos_matriz(R, elementos, opciones, guardar_img: bool = False, nombre: str = 'elementos_matriz') -> None:
    """ Graficar elementos de matriz para H2

    Parámetros
        R : distancia interatómica
        elementos : h11, h22, J11, J22, J12, K12
        guardar_img : guarda la gráfica como imagen
    """

    h11, h22, J11, J22, J12, K12 = elementos
    linewidth, fontsize, labelsize, alpha = opciones
    
    #####################
    ###### Gráfica ######
    #####################
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # 1 fila, 1 columnas, tamaño de figura (15, 5)
    
    ### Gráfica derecha ###
    #######################
    axs[0].set_rasterization_zorder(10)
    axs[0].plot(R, h11, linewidth=linewidth, linestyle='solid', label='h11', color='#ff595e')
    axs[0].plot(R, h22, linewidth=linewidth, linestyle='dashed', label='h22', color='#ff924c')
    # x config
    axs[0].set_xlim(-0.05, 7)
    axs[0].set_xlabel(r'Distancia interatómica ($a_0$)', fontsize=fontsize, labelpad=5)
    # y config
    axs[0].set_ylim(-1.8, 0.4)
    axs[0].set_ylabel(r'Energía (E$_\mathrm{h}$)', fontsize=fontsize, labelpad=10)
    axs[0].set_yticks(np.arange(-1.75, 0.3, 0.50))
    axs[0].set_yticks(np.arange(-1.75, 0.3, 0.25), minor=True)
    # plot config
    axs[0].legend(fontsize=fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[0].grid(alpha=alpha)
    
    ### Gráfica izquierda ###
    #########################
    axs[1].plot(R, J11, linewidth=linewidth, linestyle='solid', label='J11', color='#ffca3a')
    axs[1].plot(R, J22, linewidth=linewidth, linestyle='dashed', label='J22', color='#8ac926')
    axs[1].plot(R, J12, linewidth=linewidth, linestyle='dotted', label='J12', color='#1982c4')
    axs[1].plot(R, K12, linewidth=linewidth, linestyle='dashdot', label='K12', color='#6a4c93')
    # x config
    axs[1].set_xlim(-0.05, 14)
    axs[1].set_xlabel(r'Distancia interatómica ($a_0$)', fontsize=fontsize)
    # y config
    axs[1].set_ylim(0.1, 1.1)
    # plot config
    axs[1].legend(fontsize=fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[1].grid(alpha=alpha)

    ### Configuración general ###
    #############################
    fig.tight_layout(pad=2)

    if guardar_img is True:
        FILE_PATH = Path(__file__).resolve() # ruta de este archivo
        DIR_PATH = FILE_PATH.parents[2] # directorio principal
        DATA_PATH = DIR_PATH / 'imgs' # ruta de las imágenes
        formato: str = 'svg'
        path = str(DATA_PATH / f"{nombre}.{formato}")
        
        fig.savefig(path, format=f"{formato}", bbox_inches='tight')

    plt.show()