# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from pathlib import Path

import numpy as np
from numpy.typing import NDArray # type annotation
import matplotlib.pyplot as plt


def graficar_UHF(R: NDArray, y_RHF: list[NDArray], y_UHF: list[NDArray], ang: list[NDArray], EH: NDArray,
                 opciones, guardar_img: bool = False, nombre: str = 'UHF') -> None:
    """ Graficar energía UHF para H2

    Parámetros
        R : distancia interatómica
        y_RHF : [RHF1, RHF2], energías restringidas ligantes y antiligante
        y_UHF : [UHF1, UHF2], energías no restringidas ligantes y antiligante
        ang   : [angUHF1, angUHF2], ángulos de mezcla de UHF1 y UHF2 para cada distancia R
        EH : energía base del hidrógeno como referencia
        guardar_img : guarda la gráfica como imagen
    """

    RHF1, RHF2 = y_RHF
    
    if len(y_UHF) == 1:
        UHF1, UHF2 = y_UHF[0], False
    elif len(y_UHF) == 2:
        UHF1, UHF2 = y_UHF

    if len(ang) == 1:
        ang1, ang2 = ang[0], False
    elif len(ang) == 2:
        ang1, ang2 = ang

    index = next((i for i, x in enumerate(ang1) if x), None)
    x0, y0 = R[index], UHF1[index]
    x_ang, y_ang = R[index], ang1[index]
    
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))  # 1 fila, 1 columnas, tamaño de figura (15, 5)
    linewidth, fontsize, labelsize, alpha = opciones
    
    ###################
    ### Gráfica derecha
    ###################
    xmin, xmax = -0.1, 7.5
    ymin, ymax = -1.2, 0.59
    
    axs[0].axhline(y = 2*EH, color='black', linestyle='dotted') # energía base 2 átomos de hidrógeno (STO-3G)
    axs[0].plot(R, RHF1, label='RHF(+)', linestyle='solid', linewidth=linewidth, color='#a8dadc') # energías RHF orbital ligante
    axs[0].plot(R[3:], RHF2[3:], label='RHF(-)', linestyle='solid', linewidth=linewidth, color='#ffb703') # energías RHF orbital antiligante
    axs[0].plot(R, UHF1, label='UHF(+)', linestyle='dashdot', linewidth=linewidth, color='#264653') # energías UHF(+)
    if UHF2:
        axs[0].plot(R, UHF2, label='UHF(-)', linestyle='dotted', linewidth=width, color='#e63946') # energías UHF(-)
    
    axs[0].vlines(x=x0, ymin=ymin, ymax=y0, color='black', linestyle='dashed')
    axs[0].plot(x0, y0, ls="", marker="o", color='black', markersize=7) # punto a partir de donde RHF y UHF ya no son iguales
    
    # x config
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_xlabel('Distancia interatómica ($a_0$)', fontsize=fontsize, labelpad=10)
    axs[0].set_xticks(np.arange(0, xmax, 2))
    axs[0].set_xticks(np.arange(0, xmax, 1), minor=True)
    # y config
    axs[0].set_ylim(ymin, ymax)
    axs[0].set_ylabel(r'Energía ($\text{E}_\text{h}$)', fontsize=fontsize, labelpad=10)
    axs[0].set_yticks(np.arange(-1, ymax, 0.50))
    axs[0].set_yticks(np.arange(-1, ymax, 0.25), minor=True)
    # plot config
    axs[0].legend(fontsize='xx-large', handlelength=2.7)
    axs[0].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[0].grid(alpha=alpha, which='both')
    
    #####################
    ### Gráfica izquierda
    #####################
    xmin, xmax = 0, 7.5
    ymin, ymax = -0.1, 0.9
    
    axs[1].plot(R, ang1, label='UHF(+)', linestyle='dashdot', linewidth=linewidth, color='#8727da') # energías UHF
    if ang2:
        axs[1].plot(R, ang2, label='UHF(-)', linestyle='dotted', linewidth=linewidth, color='#a6c0d5') # energías UHF
    # x config
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_xlabel('Distancia interatómica ($a_0$)', fontsize=fontsize, labelpad=10)
    axs[1].set_xticks(np.arange(xmin, xmax, 2))
    axs[1].set_xticks(np.arange(xmin, xmax, 1), minor=True)
    # y config
    axs[1].set_ylim(ymin, ymax)
    axs[1].set_ylabel(r'Ángulo ($\theta$)', fontsize=fontsize, labelpad=10)
    axs[1].set_yticks(np.arange(0, 1, 0.2))
    axs[1].set_yticks(np.arange(0, 1, 0.1), minor=True)
    # plot config
    axs[1].legend(fontsize='xx-large', handlelength=2.7)
    axs[1].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[1].grid(alpha=alpha, which='both')
    
    fig.tight_layout(pad=2)
    
    if guardar_img is True:
        FILE_PATH = Path(__file__).resolve() # ruta de este archivo
        DIR_PATH = FILE_PATH.parents[2] # directorio principal
        DATA_PATH = DIR_PATH / 'imgs' # ruta de las imágenes
        formato: str = 'svg'
        path = str(DATA_PATH / f"{nombre}.{formato}")
        
        fig.savefig(path, format=f"{formato}", bbox_inches='tight')

    plt.show()