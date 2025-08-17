from template_funciones import *

# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])

def calcula_cant_aristas (A):
    # Toma una matriz de adyacencia A de un grafo no dirigido y devuelve la cantidad de aristas
    k = A.sum(axis=1).reshape(-1,1)
    return k.sum() / 2

def calcula_K (A):
    # Toma una matriz de adyacencia A de un grafo y construye la matriz diagonal K 
    grado_salida = A.sum(axis = 1) # El grado de salida de cada nodo, viendo por columna
    K = np.diag(grado_salida) # Matriz diagonal de grados de A
    return K

def simetrizar(A):
    # Dada una matriz de adyacencia A, la simetriza usando la formula dada en el enunciado
    As = np.ceil(0.5 * (A + A.T))
    return As

def mostrar_particion(particion):
    # Dada una particion (lista de listas) devuelve una tabla con los nodos de cada grupo
    datos = []
    for i, grupo in enumerate(particion):
        nombre_grupo = f"Grupo {i+1}"
        nodos = ", ".join(map(str, grupo))
        datos.append({'Grupo': nombre_grupo, 'Nodos': nodos})
    
    df = pd.DataFrame(datos)
    print(df.to_markdown(index=False))

def grafo_particion (A, particion, titulo, G_layout, barrios,ax=None):
    # Dada una matriz de adyacencia A, una partición, titulo, layout y barrios construye una visualizacion del grafo en la 
    # que se distinguen los grupos de la particion por distintos colores
    G = nx.from_numpy_array(A)

    colores = {}
    for i, grupo in enumerate(particion):
        for nodo in grupo:
            colores[nodo] = i

    color_nodos = [colores[nodo] for nodo in G.nodes()]

    creo_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,12))
        creo_fig = True

    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)
    nx.draw_networkx(G, G_layout, node_color = color_nodos, node_size=400, cmap=plt.cm.tab10, ax=ax, with_labels=False)
    nx.draw_networkx_labels(G, G_layout, font_size=6, ax=ax)
    ax.set_title(titulo)
    ax.axis("off")
    if creo_fig: plt.show()

def check_laplaciana(A):
    # Toma una matriz de adyacencia A y compara la funcion propia con la implementacion de scipy
    laplaciana_scipy = scipy.sparse.csgraph.laplacian(A)
    return np.allclose(laplaciana_scipy, calcula_L(A))

def calcula_s (v):
    # Dado un vector v, devuelve el s definido en el enunciado 
    s = np.transpose(np.sign(np.array(v)))
    return s

def checkMetPots(R, L):
    # Dadas las matrices R y L construidas a partir de una matriz de adyacencia A, se fija si los autovalores resultantes 
    # de nuestras funciones que implementan el metodo de la potencia son iguales a los devueltos por np.linalg.eig 
    autovaloresR, autovectoresR = np.linalg.eig(R)
    autovaloresL, autovectoresL = np.linalg.eig(L)

    # Ordenamos autovalores de R de mayor a menor (por valor absoluto si hay complejos)
    idx_sorted_R = np.argsort(-np.abs(autovaloresR))
    v_maxR = autovectoresR[:, idx_sorted_R[0]]
    v_maxR2 = autovectoresR[:, idx_sorted_R[1]]

    # Ordenamos autovalores de L de menor a mayor
    idx_sorted_L = np.argsort(np.abs(autovaloresL))
    v_minL0 = autovectoresL[:, idx_sorted_L[0]]
    # Si hay más de un autovalor, tomamos el segundo más chico
    v_minL = autovectoresL[:, idx_sorted_L[1]] if len(autovaloresL) > 1 else v_minL0

    # Método de la potencia
    v_max1, l_max1, _ = metpot1(R)
    v_max2, _, _ = metpot2(R, v_max1, l_max1)
    v_min_metpot = metpotI2(L, 1)[0]
    v_min_metpot0 = metpotI(L, 1)[0]

    # Comparación con abs ya que el signo no es relevante
    return (np.allclose(np.abs(v_minL.real), np.abs(v_min_metpot), atol=1e-3) and
            np.allclose(np.abs(v_maxR.real), np.abs(v_max1), atol=1e-3) and
            np.allclose(np.abs(v_maxR2.real), np.abs(v_max2), atol=1e-3) and
            np.allclose(np.abs(v_minL0.real), np.abs(v_min_metpot0), atol=1e-3))

#Funciones del template

def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    K = calcula_K(A)
    L = K - np.array(A)
    return L


def calcula_P(A):
    # Toma una matriz de adyacencia A y devuelve la matriz P 
    k = A.sum(axis=1).reshape(-1, 1) # Armamos un vector columna de grados 
    cantidad_de_aristasP = k.sum() / 2 # Pues como no es digrafo las aristas son dobles
    P = (k @ np.transpose(k)) / (2 * cantidad_de_aristasP)
    return P

def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad R
    P = calcula_P(A)
    R = np.array(A) - P
    return R

def calcula_lambda(L,v): 
    # Recibe L y v y retorna el corte asociado
    s = calcula_s(v)
    lambdon = (1/4) * (s.T @ L @ s)
    return lambdon

def calcula_Q(R,v,cantidad_de_aristas):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    s = calcula_s(v)
    Q = (1/(4 * cantidad_de_aristas)) * (s.T @ R @ s)
    return Q

def metpot1(A,tol=1e-8,maxrep=1e4):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = np.random.uniform(-1, 1, size=A.shape[1]) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v / np.linalg.norm(v, ord=2) # Lo normalizamos
   v1 = A @ v # Aplicamos la matriz una vez
   v1 = v1 / np.linalg.norm(v1, ord=2) # Normalizamos v1
   l = v @ (A @ v) # Calculamos el autovalor estimado
   l1 = v1 @ (A @ v1) # Y el estimado en el siguiente paso
   nrep = 0 
   while (np.abs(l1-l)/np.abs(l)) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # Actualizamos v y repetimos
      l = l1
      v1 = A @ v1 # Calculo nuevo v1

      norma_v1 = np.linalg.norm(v1, ord=2)
      if norma_v1 < 1e-10: return v, l, False # Check de convergencia

      v1 = v1 / norma_v1 # Normalizo
      l1 = v1 @ (A @ v1) # Calculo autovalor
      nrep += 1 
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = v @ (A @ v) # Calculamos el autovalor
   return v1,l,nrep<maxrep


def deflaciona(A,tol=1e-8,maxrep=1e4):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - (l1 * np.outer(v1, v1))
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=1e4):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeros autovectores y autovalores de A
   deflA = deflaciona(A)
   v2, l2, converge = metpot1(deflA,tol,maxrep)
   if not converge or np.linalg.norm(v2) < 1e-10: return v1, l1, False
   return v2, l2, True 
    
def metpotI(A,mu,tol=1e-8,maxrep=1e4):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    matriz_con_coef = A + mu * np.identity(A.shape[1])
    L, U = calculaLU(matriz_con_coef) # Hacemos la factorizacion Lu de la matriz para invertirla
    matriz_con_coefInv = inversa_con_lu(L, U) # Calculamos la inversa para saber su autovalor y autovector mas chico
    return metpot1(matriz_con_coefInv, tol=tol,maxrep=maxrep) # Nos devuelve el autovector y autovalor mas chicos 
                                                              # respectivamente de la matriz con shifting

def metpotI2(A,mu,tol=1e-8,maxrep=1e4):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + mu * np.identity(A.shape[1]) # Calculamos la matriz A shifteada en mu
   L, U = calculaLU(X) # Hacemos la factorizacion Lu de la matriz para invertirla              
   iX = inversa_con_lu(L, U) # La invertimos
   defliX = deflaciona(iX) # La deflacionamos
   v,l,converge =  metpot1(defliX) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu # Restamos el shift
   return v,l,converge

def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: 
        Laplaciana = calcula_L(A) # Recalculamos el L
        v,l,_ = metpotI2(Laplaciana, mu=1e-3) # Encontramos el segundo autovector de L sabiendo que con la matriz shifteada el autovector es el mismo
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        s = calcula_s(v)
        A_positiva = [i for i, valor in enumerate(v) if valor > 0]
        A_negativa = [i for i, valor in enumerate(v) if valor < 0]
        Ap = A[np.ix_(A_positiva, A_positiva)] # Asociado al signo positivo
        Am = A[np.ix_(A_negativa, A_negativa)] # Asociado al signo negativo
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    if R.shape[0] == 1: # Si llegamos al último nivel
        return [list(nombres_s)]
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return [list(nombres_s)]
        else:
            # Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            R_positiva = [i for i, valor in enumerate(v) if valor > 0]
            R_negativa = [i for i, valor in enumerate(v) if valor < 0]
            Rp = R[np.ix_(R_positiva, R_positiva)] # Parte de R asociada a los valores positivos de v
            Rm = R[np.ix_(R_negativa, R_negativa)] # Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp)  # Autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # Autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Repetimos para los subniveles
                return(
                modularidad_iterativo(Rp,Rp,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                modularidad_iterativo(Rm,Rm,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )    





