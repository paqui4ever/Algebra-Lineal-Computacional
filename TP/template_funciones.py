# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy

# Funciones dadas en el template

def construye_adyacencia(D, m):
    """ Función que construye la matriz de adyacencia del grafo de museos
        D matriz de distancias, m cantidad de links por nodo
        Retorna la matriz de adyacencia como un numpy. """
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(matriz):
    """ matriz es una matriz de NxN
        Retorna la factorización LU a través de una lista con dos matrices L y U de NxN. """
    m = matriz.shape[0]
    n = matriz.shape[1]
    Ac = matriz.copy()

    if m!=n: raise ValueError("La matriz ingresada no es cuadrada") # Devolvemos un ValueError si la matriz no es cuadrada

    for j in range(n):
        for i in range(j+1,n):
            Ac[i,j] = Ac[i,j] / Ac[j,j]
            Ac[i,j+1:] -= Ac[i,j]*Ac[j,j+1:]

    L = np.tril(Ac,-1) + np.eye(matriz.shape[0])
    U = np.triu(Ac)

    return (L, U)

def calcula_matriz_C(A):
    """
    Función para calcular la matriz de trancisiones C
    A: Matriz de adyacencia
    Retorna la matriz C
    """
    grado_salida = A.sum(axis = 1) # El grado de salida de cada nodo, viendo por columna
    Kinv = np.diag(1/grado_salida) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = np.transpose(A) @ Kinv # Calcula C multiplicando Kinv y A
    return C



def calcula_pagerank(A, alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = np.identity(N) - (1-alfa)*C
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = (alfa/N)*np.ones(N) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    np.fill_diagonal(D,1)
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = np.diag(1/F.sum(axis=1))# Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F
    C = np.transpose(F) @ Kinv # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C, cantidad_de_visitas):
    """
    Recibe la matriz C de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    suponiendo que cada visitante realizó cantidad_de_visitas pasos
    cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    """
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas - 1):
      B @= C
      B += np.eye(C.shape[0])
    return B


# Funciones propias

def graficar_grafo(D, m, alpha, barrios, Nprincipales, G_layout, ax=None):
    """
    Esta función toma el D dado, un m y alpha dados que cambiaran con lo pedido, el archivo para ver los barrios,
    un entero Nprincipales que numerará solo los N museos más relevantes y el layout para el grafo (G_layout).
    Devuelve el gráfico del grafo de museos con nodos escalados según PageRank.
    """
    # Construimos la adyacencia y calculamos PageRank
    A = construye_adyacencia(D, m)
    p = calcula_pagerank(A, alpha)
    pr = p / p.sum()  # Normalizamos como se nos dice en extra

    # Construimos el grafo
    G = nx.from_numpy_array(A)

    # Top N nodos con mayor PageRank, dado en extra
    principales = np.argsort(pr)[-Nprincipales:][::-1]
    labels = {n: str(n) if n in principales else "" for n in G.nodes}
    layout_local = {n: G_layout[n] for n in G.nodes if n in G_layout}
    colores_nodos = ['red' if n in principales else 'steelblue' for n in G.nodes]

    creo_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,12))
        creo_fig = True

    # Visualización, dado en extra
    #fig, ax = plt.subplots(figsize=(12, 12))
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)
    nx.draw_networkx(G, G_layout, node_color = colores_nodos, node_size=pr * 1e4, ax=ax, with_labels=False)
    nx.draw_networkx_labels(G, layout_local, labels=labels, font_size=6, font_color="k", ax=ax)

    ax.set_title(f"Grafo con m = {m}, alpha = {alpha}")
    ax.axis("off")

    if creo_fig: plt.show()


def calcular_ps (D, alpha, m):
    """
    Toma la matriz D, un alpha y un m y devuelve el vector p correspondiente para a matriz de adyacencia construida a partir de D y m y el alpha de la entrada
    """
    A = construye_adyacencia(D,m)
    return calcula_pagerank(A, alpha)


def tabla_valores_p(D, lista_parametros, nombre_parametro, fijo):
    """
    Toma D, una lista_parametros que serían los distintos valores de m o alpha correspondientes, el nombre del parametro que varía y luego el valor del otro parametro que quedará fijo
    Como output muestra una tabla con los vectores de PageRank p para cada valor del parámetro dado.
    """
    data = []
    for valor in lista_parametros:
      """
      Por como funciona calcular_ps, y como depende de nombre_parametro cuál es el que varía (si alpha o m), necesitamos un if para calcular el p correcto
      """
      if nombre_parametro == "alpha":
        p = calcular_ps(D,valor, fijo)
      elif nombre_parametro == "m":
        p = calcular_ps(D,fijo, valor)
      fila = {nombre_parametro: valor} # Nos armamos nuestras filas
      # Agregamos cada componente de p como una columna p_0, p_1, ..., p_n
      fila.update({f'p_{i}': round(pi, 6) for i, pi in enumerate(p)}) # Los redondeo a partir de la 6ta cifra significativa inclusive
      data.append(fila)

    df = pd.DataFrame(data)
    df.sort_values(nombre_parametro, inplace=True)

    # Mostramos la tabla
    print(f"\n Valores del vector p según variación de {nombre_parametro} (con el otro parámetro fijo en {fijo}):")
    display(df.style.set_caption("Valores de PageRank (p)").set_properties(**{"text-align": "center"}))

    return df


def tabla_top3_variando_v(variando_v, v):
    """
    Toma como entrada una lista de tuplas (p, v) donde v puede ser alpha o m, dependiendo de qué varía, y ademas una variable v
    que indica en la tabla qué variable es, si alpha o m la que cambia. 
    Con esto crea y muestra una tabla con los top 3 museos de mayor PageRank para cada v.
    """
    data = []
    for p, var in variando_v:
        top3 = list(np.argsort(p)[-3:][::-1]) # Nos armamos nuestra lista de top 3
        data.append({
            "Valor de variable": var,
            "Top 1": top3[0],
            "Top 2": top3[1],
            "Top 3": top3[2]
        })

    df = pd.DataFrame(data)
    df.sort_values("Valor de variable", inplace=True)

    # Mostrar la tabla
    print("\n Top 3 nodos según {0}: \n".format(v))
    display(df.style.set_caption("Ranking de nodos por PageRank").set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    ).set_properties(**{"text-align": "center"}))

    return df

def ranking(p):
    """
    p : Lista

    rank : lista de los indices de p ordenados de mayor a menor por el valor en esa posicion 
    position: la posicion i es la posicion de i en rank (rank[position[i]] = i)
    """

    rank = np.argsort(p)[::-1]
    position = np.empty_like(rank)
    position[rank] = np.arange(len(rank))
    return rank ,position

def graficar_top3(variacion_pageranks, valores_parametro, nombre_parametro):
    """
     Esta funcion toma como entrada una variacion_pageranks que en este caso son nuestras listas variando_m y variando_alpha,
     luego los distintos m's o alphas para los que obtuvimos ese top y por ultimo el nombre del parametro que varía
     (que aquí solamente es alpha o m) para titular correctamente nuestro gráfico
    """
    top1, top2, top3 = [], [], [] # Armamos nuestras listas de museos que son para algun alpha y m top 1, 2 y 3 respectivamente

    positions = []
    for p in variacion_pageranks: # Vamos llenando mis listas de rankings
        rank, position = ranking(p[0])

        top1.append(rank[0])
        top2.append(rank[1])
        top3.append(rank[2])

        positions.append(position)

    _,top3Graph = plt.subplots(figsize=(12, 6))
    top3Graph.plot(valores_parametro, top1, marker='o', label='Top 1')
    top3Graph.plot(valores_parametro, top2, marker='s', label='Top 2')
    top3Graph.plot(valores_parametro, top3, marker='^', label='Top 3')

    for i, x in enumerate(valores_parametro):
        top3Graph.annotate(str(top1[i]), (x, top1[i] + 0.2), ha='center', fontsize=9)
        top3Graph.annotate(str(top2[i]), (x, top2[i] + 0.2), ha='center', fontsize=9)
        top3Graph.annotate(str(top3[i]), (x, top3[i] + 0.2), ha='center', fontsize=9)
    
    top3Graph.set_xlabel(nombre_parametro)
    top3Graph.set_ylabel('Numeración del museo')
    top3Graph.set_title(f'Variación del Top 3 de PageRank según {nombre_parametro}') # Ponemos el respectivo titulo
    top3Graph.grid(True)

    madeItTo3 = set(top1+top2+top3)
    
    _,rankGraph = plt.subplots(figsize=(12, 6))

    for m in madeItTo3:
        rankGraph.plot(valores_parametro, list(map(lambda l: l[m],positions)), label=m)
    
    rankGraph.invert_yaxis()
    
    rankGraph.set_xlabel(nombre_parametro)
    rankGraph.set_ylabel('Posicion en el Ranking')
    rankGraph.set_title(f'Variación del ranking de los museos que llegaron al Top 3')
    rankGraph.grid(True)
    
    plt.legend()
    plt.show()


def calcular_normaMatricial1 (M):
  """
  Dada una matriz M, calcula la norma 1 de la misma
  """
  M = M.copy()
  M = np.vectorize(abs)(M) # Aplica la funcion abs (modulo) a todos los elementos de la matriz
  sum_cols = M.sum(axis=0) # Calcula la suma de las columnas
  return max(sum_cols) # Devuelve el maximo de las sumas


def inversa_con_lu(L, U):
    """
    Calcula la inversa de una matriz A dada su descomposición LU. Toma una L matriz triangular inferior de la factorización LU 
    y U otra matriz triangular superior de la factorización LU.
    Devuelve Ai, la matriz inversa de A
    """
    n = L.shape[0]
    Ai = np.zeros((n, n))
    I = np.eye(n)

    for i in range(n):
        y = scipy.linalg.solve_triangular(L, I[:, i], lower=True, unit_diagonal=True) #Resolver L y = e_i
        x = scipy.linalg.solve_triangular(U, y, lower=False) #Resolver U x = y
        Ai[:, i] = x # Colocar x como la i-ésima columna de la inversa

    return Ai

def numero_de_condicion (M):
  """
  Dada una matriz M calcula el número de condicion con norma 1 de la misma
  """
  L,U = calculaLU(M)
  Mi = inversa_con_lu(L,U) # calcula el inverso de M usando LU
  return calcular_normaMatricial1(M) * calcular_normaMatricial1(Mi)
