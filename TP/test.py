import pandas as pd 
import networkx as nx 
import random
import matplotlib.pyplot as plt

def modificarTamaño (n) -> int:
    numeroRandom = random.randint(1, 100)
    n += numeroRandom
    return n

nodes = [1,2,3,4]
edges = [(i, i+1) for i in range (1, len(nodes))]
labels = {node: node for node in nodes}
print(edges)
print(labels)

graph = nx.Graph()
graph.add_edges_from(edges)
graph.add_nodes_from(nodes)
node_sizes = [node * 100 for node in graph.nodes()] # Los multiplico por 100 para que se vean
node_sizes2 = [modificarTamaño(node)*100 for node in graph.nodes()] # Prueba para ver si uso funciones como forma de modificar los tamaños de los nodos

nx.draw(graph, pos = nx.spring_layout(graph), node_size = node_sizes2, labels=labels, with_labels=True)   # Uso spring layout para que la longitud de las aristas represente la distancia entre nodos
plt.show()



def graficar_grafo (D, m, alpha):
  matriz_para_Page = construye_adyacencia(D, m) # Mi matriz de adyacencia para luego aplicarle page rank
  matriz_con_Page_aplicado = calcula_pagerank(matriz_para_Page, alpha) # Me devuelve mi p con mis valores de pagerank para todo museo
  grafo = nx.from_numpy_array(matriz_para_Page)

  """
    pageranks = np.array(matriz_con_Page_aplicado)
    normalizados = (pageranks - pageranks.min()) / (pageranks.max() - pageranks.min())
    tamaño_nodos = [50 + 100 * p for p in normalizados]

  """
  #matriz2Page = nx.pagerank(grafo, alpha=0.85) solo para ver si dan resultados distintos
  #print(matriz2Page)

  tamaño_nodos = [p*10000 for p in matriz_con_Page_aplicado]
  fig, ax = plt.subplots(figsize=(15, 15)) # Visualización de la red en el mapa
  barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
  nx.draw_networkx(grafo, G_layout, node_size = tamaño_nodos, ax = ax)
  plt.show()

def graficar_grafo (D, m, alpha):
  matriz_adyacencia = construye_adyacencia(D, m) # Mi matriz de adyacencia para luego aplicarle page rank
  p = calcula_pagerank(matriz_adyacencia, alpha) # Me devuelve mi p con mis valores de pagerank para todo museo
  grafo = nx.from_numpy_array(matriz_adyacencia) # Construyo mi grafo

  print(p)

  tamaño_nodos = [pi*10000 for pi in p]
  fig, ax = plt.subplots(figsize=(15, 15)) # Visualización de la red en el mapa
  barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
  nx.draw_networkx(grafo, G_layout, node_size = tamaño_nodos, ax = ax)
  plt.show()

graficar_grafo(D, 3, (1/5))

factor_escala = 1e4 # Escalamos los nodos 10 mil veces para que sean bien visibles
fig, ax = plt.subplots(figsize=(10, 10)) # Visualización de la red en el mapa
barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
pr = np.random.uniform(0,1,museos.shape[0])# Este va a ser su score Page Rank. Ahora lo reemplazamos con un vector al azar
pr = pr/pr.sum() # Normalizamos para que sume 1
Nprincipales = 5 # Cantidad de principales
principales = np.argsort(pr)[-Nprincipales:] # Identificamos a los N principales
labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos
nx.draw_networkx(G,G_layout,node_size = pr*factor_escala, ax=ax,with_labels=False) # Graficamos red
nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k") # Agregamos los nombres

variando_m = [] # Util para la segunda parte del ejercicio
for m in [1,3,5,10]:
  p = calcular_ps(1/5, m)
  variando_m.append((p,m)) # Guardamos cada p obtenido con su m que varió correspondiente
  print("Para m = {0} tengo un p = {1}".format(m, p))

variando_alpha = [] # Será usado para la segunda parte del ejercicio
for alpha in [6/7, 4/5, 2/3, 1/2, 1/3, 1/5, 1/7]:
  p = calcular_ps(alpha, 5)
  variando_alpha.append((p, alpha)) # Guardamos cada p calculado con su alpha que varió correspondiente
  print("Para alpha = {0} tengo un p = {1}".format(alpha, p))

def graficar_top3_en_grafo(D, p, m, G_layout, titulo=''): # Funcion que posiblemente sea borrada despues
    G = nx.from_numpy_array(construye_adyacencia(D,m))
    top3 = np.argsort(p[0])[-3:][::-1]

    num_nodos = len(p[0])

    colores = []
    tamaños = []

    for i in range(num_nodos):
        if i == top3[0]:
            colores.append('red')
            tamaños.append(900)
        elif i == top3[1]:
            colores.append('orange')
            tamaños.append(700)
        elif i == top3[2]:
            colores.append('green')
            tamaños.append(600)
        else:
            colores.append('skyblue')
            tamaños.append(300)

    plt.figure(figsize=(12, 12))
    nx.draw_networkx(G, pos=G_layout, node_size=tamaños, node_color=colores, with_labels=False)

    # Agregar etiquetas solo a los top 3
    nx.draw_networkx_labels(G, pos=G_layout, font_size=10, font_color='black')

    #plt.title(titulo)
    plt.axis('off')
    plt.show()

#graficar_top3_en_grafo(D, variando_alpha[3], 3, G_layout, titulo=f'Top 3 para alpha={variando_alpha[2]}')
#graficar_top3_en_grafo(D, variando_m[1], 3, G_layout, "Top 3 para m = 3")

def graficar_top3_en_grafo2(D, p, m, G_layout, barrios, titulo=''):
    """
    Esta funcion toma una D (matriz de distancias), un vector p de PageRank,
    el m para calcular la matriz de adyacencia, el layout de los nodos dado más arriba, el archivo con los barrios y por ultimo un titulo.
    Devuelve un grafico de CABA acorde a los parametros ingresados en el que se le pone la label unicamente a los 3 nodos del top 3.
    """
    G = nx.from_numpy_array(construye_adyacencia(D, m))
    factor_escala = 1e4  # Escala para el tamaño de nodos
    fig, ax = plt.subplots(figsize=(10, 10))
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)
    pr = p[0] / p[0].sum()

    # Top 3 nodos por PageRank
    top3 = np.argsort(pr)[-3:][::-1] # cambiar

    # Etiquetas: solo los top 3 con su índice
    labels = {n: str(n) if n in top3 else "" for n in G.nodes} # cambiar luego de hacer lo de arriba

    # Graficamos el grafo
    nx.draw_networkx(G, G_layout, node_size=pr * factor_escala, ax=ax, with_labels=False)
    nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=8, font_color="black")

    plt.title(f"{titulo}")
    plt.axis('off')
    plt.show()

barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')

graficar_top3_en_grafo2(D, variando_alpha[3], 3, G_layout, barrios, "Top 3 para alpha")
graficar_top3_en_grafo2(D, variando_m[1], 3, G_layout, barrios, "Top 3 para m = 3")

def calcular_normaMatricial_uno (M):
  """
  Dada una matriz M, calcula la norma 1 de la misma
  """
  M = M.copy()
  M = np.vectorize(abs)(M) # Aplica la funcion abs (modulo) a todos los elementos de la matriz
  sum_cols = M.sum(axis=0) # Calcula la suma de las columnas
  return max(sum_cols) # Devuelve el maximo de las sumas

def inversa_de_matriz (M):
  """
  Dada una matriz M, calcula su inversa
  """
  N = M.shape[0]
  L,U = calculaLU(M)
  I = np.eye(N)
  res = []
  for i in range(N):
    e = I[i,:]
    Up = scipy.linalg.solve_triangular(L,e,lower=True)
    p = scipy.linalg.solve_triangular(U,Up)
    res.append(p)
  return np.array(res)

def numero_de_condicion (M):
  """
  Dada una matriz M calcula el número de condicion con norma 1 de la misma
  """
  L,U = calculaLU(M)
  Mi = np.linalg.inv(M) # calcula el inverso de M usando LU
  return calcular_normaMatricial_uno(M) * calcular_normaMatricial_uno(Mi)

C = calcula_matriz_C_continua(D)
B = calcula_B(C,3)
num_cond = numero_de_condicion(B)
num_cond_np = np.linalg.cond(B,p=1)

error = a1 - (num_cond/num_cond_np)
print(num_cond,num_cond_np)
print("{:.6f}".format(error)+"%") #diferencia entre nuestra implementacion del numero de condicion y la de numpy para B