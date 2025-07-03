# Script para el cálculo de elementos de uno y dos cuerpos
# Molécula de hidrógeno STO-3G

from g_pqrs import *
from extras import *

def un_cuerpo(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray, ZA:int, ZB:int) -> list:
    """ Integrales de un solo cuerpo para H2 con la base STO-3G

    Parámetros
        d: vector de coeficientes de contracción
        a: vector de exponentes orbitales gaussianos
        (RA, RB): coord. del núcleo (A, B)
        (ZA, ZB): carga del núcleo (A, B)
    """
    f11 = fpp(1, d, a, RA, RB, ZA, ZB) # h11 energía cinética (orbital ligante)
    f33 = fpp(3, d, a, RA, RB, ZA, ZB) # h22 energía cinética (orbital antiligante)
    return f11, f33


def dos_cuerpos(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray, ZA:int, ZB:int) -> list:
    """ Integrales de dos cuerpos para H2 con la base STO-3G

    Parámetros
        d: vector de coeficientes de contracción
        a: vector de exponentes orbitales gaussianos
        (RA, RB): coord. del núcleo (A, B)
        (ZA, ZB): carga del núcleo (A, B)
    """
    g1212 = gpqrs([1,2,1,2], d, a, RA, RB) # término Coulombiano J11
    g3434 = gpqrs([3,4,3,4], d, a, RA, RB) # término Coulombiano J22
    g1313 = gpqrs([1,3,1,3], d, a, RA, RB) # término Coulombiano J12
    g1331 = gpqrs([1,3,3,1], d, a, RA, RB) # término Coulombiano K12 (intercambio)
    return g1212, g3434, g1313, g1331


def elementos_matriz(d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray, ZA:int, ZB:int) -> list:
    """ Elementos de matriz para H2 con la base STO-3G

    Parámetros
        d: vector de coeficientes de contracción
        a: vector de exponentes orbitales gaussianos
        (RA, RB): coord. del núcleo (A, B)
        (ZA, ZB): carga del núcleo (A, B)
    """
    # elementos de un cuerpo
    f11, f33 = un_cuerpo(d, a, RA, RB, ZA, ZB)
    # elementos de dos cuerpos
    g1212, g3434, g1313, g1331 = dos_cuerpos(d, a, RA, RB, ZA, ZB)
    
    return f11, f33, g1212, g3434, g1313, g1331


def elementos_matriz_distancias(inicio:float, fin:float, paso:float, d:np.ndarray, a:np.ndarray, ZA:int, ZB:int) -> list:
    """ Lista de elementos de matriz de H2 con la base STO-3G para
    diversas distancias internucleares en el intervalo (inicio, fin)
    con incrementos de 'paso' 

    Parámetros
        (inicio, fin) : separación interatómica inicial (final) dada en u.a.
        paso : tamaño del incremento en la separación interatómica
        d : vector de coeficientes de contracción
        a : vector de exponentes orbitales gaussianos
        (ZA, ZB): carga del núcleo (A, B)
    """
    h11, h22, J11, J22, J12, K12, hnuc = [], [], [], [], [], [], []

    distancias = np.arange(inicio, fin, paso) # distancias internucleares
    decimal = len(str(paso).split('.')[1]) # número de decimales en el paso
    distancias = np.round(distancias, decimal) # remover error numérico redondeando al número de decimales previsto
    # En esto último se asume que 'inicio', 'fin' y 'paso' tienen a lo más el mismo número de cifrar decimales
    
    for x in distancias:
        RA = np.array([0, 0, 0]) # núcleo A fijo en 0
        RB = np.array([x, 0, 0]) # posición del núcleo B
        h11_val, h22_val, J11_val, J22_val, J12_val, K12_val = elementos_matriz(d, a, RA, RB, ZA, ZB)
        hnuc_val = 1/(np.linalg.norm(RA-RB)) # repulsión nuclear
        h11.append(h11_val)
        h22.append(h22_val)
        J11.append(J11_val)
        J22.append(J22_val)
        J12.append(J12_val)
        K12.append(K12_val)
        hnuc.append(hnuc_val)
    
    # convertir listas en arreglos de numpy
    h11, h22, J11, J22, J12, K12, hnuc = [np.array(h11), np.array(h22), np.array(J11), np.array(J22), np.array(J12), np.array(K12), np.array(hnuc)]

    return distancias, h11, h22, J11, J22, J12, K12, hnuc


def elementos_H2(modo: str = 'cargar', intervalo: list = None):
    """ Cargar o calcular elementos de matriz para H2

    Parámetros
        modo:
            'cargar' -> emplear los elementos de matriz de los archivos del directorio 'data'
            'calcular' -> calcular los elementos de matriz
        intervalo: [inicio, fin, paso]
            indica la distancia interatómica de inicio, fin y paso a considerar para calcular los elementos de matriz
            los elementos de matriz se guardan como .csv en el directorio 'data' bajo el nombre 'H2_nombre'
    """
    # cargar elementos de matriz h11, h22, J11, J22, J12, K12 y hnuc
    if modo == 'cargar':
        distancias = np.loadtxt("data/H2_distancias.csv", delimiter=",", dtype=('float'), skiprows=1)
        h11 = np.loadtxt("data/H2_h11.csv", delimiter=",", dtype=('float'), skiprows=1)
        h22 = np.loadtxt("data/H2_h22.csv", delimiter=",", dtype=('float'), skiprows=1)
        J11 = np.loadtxt("data/H2_J11.csv", delimiter=",", dtype=('float'), skiprows=1)
        J22 = np.loadtxt("data/H2_J22.csv", delimiter=",", dtype=('float'), skiprows=1)
        J12 = np.loadtxt("data/H2_J12.csv", delimiter=",", dtype=('float'), skiprows=1)
        K12 = np.loadtxt("data/H2_K12.csv", delimiter=",", dtype=('float'), skiprows=1)
        hnuc = np.loadtxt("data/H2_hnuc.csv", delimiter=",", dtype=('float'), skiprows=1)
        print("Elementos de matriz y distancias interatómicas cargadas.")
    # calcular elementos de matriz h11, h22, J11, J22, J12, K12 y hnuc
    elif modo == 'calcular':
        inicio, fin, paso = intervalo
        distancias, h11, h22, J11, J22, J12, K12, hnuc = elementos_matriz_distancias(inicio, fin, paso, d, a, ZA, ZB)
        print('Elementos de matriz calculados.')
        # guardar
        guardar('H2_distancias', distancias)
        guardar('H2_h11', h11)
        guardar('H2_h22', h22)
        guardar('H2_J11', J11)
        guardar('H2_J22', J22)
        guardar('H2_J12', J12)
        guardar('H2_K12', K12)
        guardar('H2_hnuc', hnuc)
        print("Elementos de matriz y distancias interatómicas guardadas en el directorio 'data'.")
        
    return distancias, h11, h22, J11, J22, J12, K12, hnuc