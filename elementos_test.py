import numpy as np
from scipy.special import erf
from math import floor, log10

# Segura
# d = np.array([0.44471812476789035, 0.5352716544572346, 0.1543000507808527]) # coeficientes de contracción
# a = np.array([0.16887939463273338, 0.6240343336327064, 3.4256944279866635]) # exponentes orbitales Gaussianos

# Stewart
d = np.array([0.444635, 0.535328, 0.154329])
a = np.array([0.109818, 0.405771, 2.22766])

# distancia interatómica de 1.401 u.a.
RA = np.array([0, 0, 0])
RB = np.array([1.401, 0, 0])

# carga nuclear
ZA, ZB = 1, 1

#############################################
########## FACTORES
#############################################

def GaussNorm(a : int):
    """ Factor de normalización Gaussiano 1s
    a : exponente Gaussiano
    """
    return np.power(2*a/np.pi, 3/4)

def arg(a : float, b : float, RA : np.array, RB : np.array):
    """ Argumento del factor pre-exponencial K
    
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    p = a+b # exponente total
    mu = a*b/p
    RAB2 = np.square(np.linalg.norm(RA-RB))
    return -mu*RAB2

def K1(a : float, b : float, RA : np.array, RB : np.array):
    """ Factor pre-exponencial (integral 1 cuerpo)
    
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    return np.exp(arg(a, b, RA, RB))

def K2(a : float, b : float, c : float, d : float, RA : np.array, RB : np.array, RC : np.array, RD : np.array):
    """ Factor pre-exponencial (integral 2 cuerpos)
    
    (a, b, c, d) : exponente orbital Gaussiano
    (RA, RB, RC, RD) : coordenada del núcleo (A, B, C, D)
    """
    return np.exp(arg(a, c, RA, RC) + arg(b, d, RB, RD))

#############################################
########## TRASLAPE
#############################################

def Spq(a : float, b : float, RA : np.array, RB : np.array):
    """ Integral de traslape S_pq (normalizada)
    
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    return GaussNorm(a) * GaussNorm(b) * np.power(np.pi/(a+b), 3/2) * K1(a, b, RA, RB)

def Smn(d : np.array, a : np.array, RA : np.array, RB : np.array):
    """ Integral de traslape total S_mn
    
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB) : coordenada del núcleo (A, B)
    """
    Mmn = 0 # elemento de matriz
    k = len(d)
    for p in range(k):
        for q in range(k):
            Mmn += d[p] * d[q] * Spq(a[p], a[q], RA, RB) # elemento de matriz
    return Mmn

SAA = Smn(d, a, RA, RA) # elemento de matriz S_AA y S_BB

# cifra decimal a la cual se redondearán los elementos de matriz S
# la cifra será del orden de magnitud del primer dígito diferente de cero para la condición de normalización
vdecimal = abs(floor(log10(abs(float(SAA) % 1)))+1)

#############################################
########## CINÉTICA
#############################################

def Tpq(a : np.array, b : np.array, RA : np.array, RB : np.array):
    """ Integral cinética S_mn (normalizada)
    
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB) : coordenada del núcleo (A, B)
    """
    RAB2 = np.square(np.linalg.norm(RA-RB))
    return (a*b)/(a+b) * (3 - 2*(a*b)/(a+b)*RAB2) * Spq(a, b, RA, RB)

def Tmn(d : np.array, a : np.array, RA : np.array, RB : np.array):
    """ Integral cinética total S_mn
    
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB) : coordenada del núcleo (A, B)
    """
    Mmn = 0 # elemento de matriz
    k = len(d)
    for p in range(k):
        for q in range(k):
            Mmn += d[p] * d[q] * Tpq(a[p], a[q], RA, RB) # elemento de matriz
    return np.round(Mmn, 4)

#############################################
########## COULOMBIANA
#############################################

def F0(t : float):
    """ Función de Boys, n=0
    
    t : argumento de la función
    """
    return (1/2) * np.sqrt(np.pi/t) * erf(np.sqrt(t))

def RP(a : float, b : float, RA : np.array, RB : np.array):
    """ Coordenada de centro de carga
    
    (a, b) : exponente orbital Gaussiano
    (RA, RB) : coordenada del núcleo (A, B)
    """
    p = a + b # exponente total
    return (a*RA+b*RB)/p

def Vpq_AB(a : float, b : float, RA : np.array, RB : np.array, RC : np.array, ZC : float):
    """ Integral coulombiana V_pq
    
    (a, b) : exponente orbital Gaussiano
    (RA, RB, RC) : coordenada del núcleo (A, B, C)
    ZC : carga del núcleo C
    """
    factor = -2*np.pi/(a+b) * ZC # factor común del término pq
    
    if np.array_equal(RA, RB) and np.array_equal(RB, RC): # todos los núcleos iguales
        Vpq = factor
    else: # cualquier otro caso
        RAB2 = np.square(np.linalg.norm(RA-RB))
        Rp = RP(a, b, RA, RB) # coordenada de centro de carga
        RPC2 = np.square(np.linalg.norm(Rp-RC))
        Vpq = factor * K1(a, b, RA, RB) * F0((a+b)*RPC2)
    return Vpq

def Vmn1(d : np.array, a : np.array, RA : np.array, RB : np.array, RC : np.array, ZC : float):
    """ Integral coulombiana total V_mn
    
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB, RC) : coordenada del núcleo (A, B, C)
    ZC : carga del núcleo C
    """
    Mmn1 = 0 # elemento de matriz V^i_AA
    Mmn2 = 0 # elemento de matriz V^i_AB y V^i_BA
    Mmn3 = 0 # elemento de matriz V^i_BB
    k = len(d)

    for p in range(k):
        for q in range(k):
            factor = d[p] * d[q] * GaussNorm(a[p]) * GaussNorm(a[q]) # factor común del término p, q
            Mmn1 += factor * Vpq_AB(a[p], a[q], RA, RA, RC, ZC)
            Mmn2 += factor * Vpq_AB(a[p], a[q], RA, RB, RC, ZC)
            Mmn3 += factor * Vpq_AB(a[p], a[q], RB, RB, RC, ZC)
    
    return np.round([Mmn1, Mmn2, Mmn3], 4) # elementos de matriz (V^i_AA, V^i_AB, V^i_BB)

#############################################
########## DOS ELECTRONES
#############################################

def ACBD(a : float, b : float, c : float, d : float, RA : np.array, RB : np.array, RC : np.array, RD : np.array):
    """ Integral de dos electrones (AC|BD)
    
    (a, b, c, d) : exponentes orbitales Gaussianos
    (RA, RB, RC, RD) : coordenadas del núcleo (A, B, C, D)

        A, C : asociados al electrón 1
        B, D : asociados al electrón 2
    """

    factor = 2*np.power(np.pi, 5/2) / ( (a+c) * (b+d) * np.sqrt(a+b+c+d) )

    if np.array_equal(RA, RB) and np.array_equal(RB, RC) and np.array_equal(RC, RD):
        term = factor
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

def V12(a : float, b : float, c : float, d : float, RA : np.array, RB : np.array, RC : np.array, RD : np.array):
    """ Integral de dos electrones total (normalizada)
        
    (a, b, c, d) : exponentes orbitales Gaussianos
    (RA, RB, RC, RD) : coordenadas del núcleo (A, B, C, D)

        A, C : asociados al electrón 1
        B, D : asociados al electrón 2
    """
    v12 = GaussNorm(a) * GaussNorm(b) * GaussNorm(c) * GaussNorm(d) * ACBD(a, b, c, d, RA, RB, RC, RD)
    return v12

def Vmn2(d : np.array, a : np.array, RA : np.array, RB : np.array, RC : np.array, RD : np.array):
    """ Elemento de matriz de interacción de dos electrones

    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB, RC, RD) : coordenada del núcleo (A, B, C, D)
    """
    Mijkl = 0 # elemento de tensor
    L = len(d)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for l in range(L):
                    Mijkl += d[i] * d[j] * d[k] * d[l] * V12(a[i], a[j], a[k], a[l], RA, RB, RC, RD)
    return np.round(Mijkl, 4)

#############################################
########## COEF. DE NORMALIZACIÓN
#############################################

def cPM(d: np.array, a: np.array, RA: np.array, RB: np.array, signo : int):
    """ Constante de normalización c+-
            Psi = c+- (Phi_A +- Phi_B)
            
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB) : coordenada del núcleo (A, B)
    signo : signo de S en la normalización
         1 -> positivo
        -1 -> negativo
    """
    S = Smn(d, a, RA, RB)
    ord_mag = np.power(10,vdecimal)
    S = np.trunc(S*ord_mag)/ord_mag # truncar S hasta el primer decimal diferente de cero para S_AA
    return np.round(1/np.sqrt(2*(1+signo*S)), 4)

#############################################
########## f_pq
#############################################
        
def fpp(p : int, d : np.array, a : np.array, RA : np.array, RB : np.array, ZA : float, ZB : float):
    """ Elemento de matriz f_pp
    
    p : elemento de la base, base = {X_1, X_2, X_3, X_4}
        p = 1, 2, 3, 4
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB) : coordenada del núcleo (A, B)
    (ZA, ZB) : carga del núcleo (A, B)
    """
    TAA = Tmn(d, a, RA, RA) # elemento de matriz T_AA y T_BB
    TAB = Tmn(d, a, RA, RB) # elemento de matriz T_AB y T_BA
    V1_AA, V1_AB, V1_BB = Vmn1(d, a, RA, RB, RA, ZA) # elementos de matriz V^1_AA, V^1_AB y V^1_BB
    V2_AA, V2_AB, V2_BB = Vmn1(d, a, RA, RB, RB, ZB) # elementos de matriz V^2_AA, V^2_AB y V^2_BB

    sgn = 1 if p in [1,2] else -1 # signo de p
    c2 = np.square(cPM(d, a, RA, RB, sgn))
    term = c2*( 2*(TAA + sgn*TAB) + (V1_AA + V1_BB) + (V2_AA + V2_BB) + 2*sgn*(V1_AB + V2_AB) )
    return term

#############################################
########## g_pqrs
#############################################

def gpqrs(P : np.array, d : np.array, a : np.array, RA : np.array, RB : np.array):
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