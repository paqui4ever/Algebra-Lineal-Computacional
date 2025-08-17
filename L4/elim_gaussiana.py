#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    for col in range (len(Ac)): # Itero columna por columna
        for fila in range (col+1,len(Ac)): # Itero fila por fila
            coef = Ac[fila][col] / Ac[col][col] # Veo el coeficiente por el que voy a multiplicar el elemento de la diag
            for k in range (len(Ac[0])):
                Ac[fila][k] -= coef * Ac[col][k]

    print("esto es AC: \n", Ac)

    L = np.tril(Ac,-1) + np.eye(A.shape[0]) 
    U = np.triu(Ac)

    def solve_L (L,y):
        return L @ y 
    
    def solve_U (U,x):
        return U @ x
                
    ## hasta aqui
    
    return L, U, cant_op


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()

A = [[2,1,2,3],[4,3,3,3],[-2,2,-4,-12],[4,1,8,-3]]
    
elim_gaussiana(np.array(A))
    