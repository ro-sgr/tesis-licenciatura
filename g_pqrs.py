# Script para el cálculo de elementos de dos cuerpos
# Molécula de hidrógeno STO-3G

from f_pq import *

# Referencias.
# [1] A. Szabo y N. S. Ostlund. Modern Quantum Chemistry. Introduction to Advanced Electronic Structure Theory. Dover Publications, 1989.
# [2] T. Helgaker, P. Jørgensen y J. Olsen. Molecular Electronic-Structure Theory. John Wiley & Sons, Ltd, 2000.

#############################################
########## FACTORES
#############################################

def K2(a:float, b:float, c:float, d:float, RA:np.ndarray, RB:np.ndarray, RC:np.ndarray, RD:np.ndarray) -> np.float64:
    """ Factor pre-exponencial (integral 2 cuerpos)
    
    (a, b, c, d) : exponente orbital Gaussiano
    (RA, RB, RC, RD) : coordenada del núcleo (A, B, C, D)
    """
    return np.exp(arg(a, c, RA, RC) + arg(b, d, RB, RD))

#############################################
########## DOS ELECTRONES
#############################################

def ACBD(a:float, b:float, c:float, d:float, RA:np.ndarray, RB:np.ndarray, RC:np.ndarray, RD:np.ndarray):
    """ Integral de dos electrones (AC|BD)
    
    (a, b, c, d) : exponentes orbitales Gaussianos
    (RA, RB, RC, RD) : coordenadas del núcleo (A, B, C, D)

        A, C : asociados al electrón 1
        B, D : asociados al electrón 2
    """

    factor = 2*np.power(np.pi, 5/2) / ( (a+c) * (b+d) * np.sqrt(a+b+c+d) )

    if np.array_equal(RA, RB) and np.array_equal(RB, RC) and np.array_equal(RC, RD):
        return factor
    else:
        Rp = RP(a, c, RA, RC)
        Rq = RP(b, d, RB, RD)
        RPQ2 = np.square(np.linalg.norm(Rp-Rq))
    
        if (np.array_equal(RA, RB) and np.array_equal(RC, RD)) and ( (a==b and c==d) or (a==c and b==d) or (a==d and b==c) ):
            term = factor * K2(a, b, c, d, RA, RB, RC, RD)
        elif (np.array_equal(RA, RD) and np.array_equal(RB, RC)) and ( (a==b and c==d) or (a==c and b==d) or (a==d and b==c) ):
            term = factor * K2(a, b, c, d, RA, RB, RC, RD)
        elif (np.array_equal(RA, RC) and np.array_equal(RB, RD)) and ( (a==b and c==d) or (a==c and b==d) or (a==d and b==c) ):
            term = factor * F0((a+c)*(b+d)/(a+b+c+d)*RPQ2)
        else:
            term = factor * K2(a, b, c, d, RA, RB, RC, RD) * F0((a+c)*(b+d)/(a+b+c+d)*RPQ2)
        return term

def V12(a:float, b:float, c:float, d:float, RA:np.ndarray, RB:np.ndarray, RC:np.ndarray, RD:np.ndarray):
    """ Integral de dos electrones total (normalizada)
        
    (a, b, c, d) : exponentes orbitales Gaussianos
    (RA, RB, RC, RD) : coordenadas del núcleo (A, B, C, D)

        A, C : asociados al electrón 1
        B, D : asociados al electrón 2
    """
    v12 = GaussNorm(a) * GaussNorm(b) * GaussNorm(c) * GaussNorm(d) * ACBD(a, b, c, d, RA, RB, RC, RD)
    return v12

def Vmn2(d :np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray, RC:np.ndarray, RD:np.ndarray):
    """ Elemento de matriz de interacción de dos electrones

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB, RC, RD) : coordenadas del núcleo (A, B, C, D)
    """
    Mijkl = 0 # elemento de tensor
    L = len(d)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for l in range(L):
                    Mijkl += d[i] * d[j] * d[k] * d[l] * V12(a[i], a[j], a[k], a[l], RA, RB, RC, RD)
    return Mijkl

#############################################
########## g_pqrs
#############################################

def gpqrs(P:np.ndarray, d:np.ndarray, a:np.ndarray, RA:np.ndarray, RB:np.ndarray):
    """ Elemento de matriz g_pqrs
    
    P : vector de elementos de la base, P = (p,q,r,s)
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB) : coordenada del núcleo (A, B)
    """
    p,q,r,s = P
    nucleos = [RA, RB]

    Mpqrs = 0 # elemento de tensor

    # si ocurre que (p,r) o (q,s) no son simultáneamente par o impar, g_pqrs = 0 (debido al espín)
    if ((p in [1,3] and r not in [1,3]) or (p in [2,4] and r not in [2,4])) or ((q in [1,3] and s not in [1,3]) or (q in [2,4] and s not in [2,4])):
        Mpqrs = 0 # elemento de tensor
    else:
        sgn_p = 1 if p in [1,2] else -1 # signo de p
        sgn_q = 1 if q in [1,2] else -1 # signo de q
        sgn_r = 1 if r in [1,2] else -1 # signo de r
        sgn_s = 1 if s in [1,2] else -1 # signo de s
    
        # coeficientes de normalización (p,q,r,s)
        cp = cPM(d, a, RA, RB, sgn_p)
        cq = cPM(d, a, RA, RB, sgn_q)
        cr = cPM(d, a, RA, RB, sgn_r)
        cs = cPM(d, a, RA, RB, sgn_s)
        
        for A in nucleos:
            for B in nucleos:
                for C in nucleos:
                    for D in nucleos:
                        sign = 1
                        if np.array_equal(A, RB):
                            sign *= sgn_p
                        if np.array_equal(B, RB):
                            sign *= sgn_q
                        if np.array_equal(C, RB):
                            sign *= sgn_r
                        if np.array_equal(D, RB):
                            sign *= sgn_s

                        mpqrs = sign*Vmn2(d, a, A, B, C, D)
                        Mpqrs += mpqrs
    
        Mpqrs = cp*cq*cr*cs*Mpqrs # multiplicar Mpqrs por coefs. de normalización
        
    return Mpqrs