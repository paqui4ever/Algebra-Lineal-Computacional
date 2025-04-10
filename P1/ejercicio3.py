import numpy as np 

a = 7 
b = a + 1
print("b = ", b)

#Vectores

v = np.array ([1,2,3,4])
w = np.array ([2,3,4,0])

print("v + w = ", v+w)
print("2*v = ", 2*v)
print("v**2 = ", v**2)


# Matrices

A = np.array([[1,2,3,4,5], [0,1,2,3,4], [2,3,4,5,6], [0,0,1,2,3], [0,0,0,0,1]])
print(A)
print(A[0:2, 3:5]) # Elije las filas 0 a 2 exclusive, columnas 3 a 5 exclusive
print(A[:2, 3:]) # Otra forma de escribir lo de arriba
print(A[[0,2,4],:]) # Agarra las filas 0,2 y 4 completas
indices = np.array([0,2,4])
print(A[indices,indices]) # Indexacion avanzada: Elije fila 0 columna 0, fila 2 columna 2 y fila 4 columna 4
print(A[indices, indices[:, None]]) # Hace la lista indices como columna y hace producto cruzado de indices
                                    # Entonces termino teniendo las 3^2 posibles combinaciones entre indices
                                    # Asi hace (0,0), (0,2), (0,4) y asi con el 2 y 4 tambien