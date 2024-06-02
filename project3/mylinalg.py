"""

Functions to solve linear systems of equations.

Kuo-Chuan Pan
2024.05.05

"""

import numpy as np


def solveLowerTriangular(L, b):
    """
    Solve a linear system with a lower triangular matrix L.

    Arguments:
    L -- a lower triangular matrix
    b -- a vector

    Returns:
    x -- the solution to the linear system
    """
    n = len(b)
    x = np.zeros(n)

    # TODO
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    return x


def solveUpperTriangular(U, b):
    """
    Solve a linear system with an upper triangular matrix U.

    Arguments:
    U -- an upper triangular matrix
    b -- a vector

    Returns:
    x -- the solution to the linear system

    """
    n = len(b)
    x = np.zeros(n)

    # TODO
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]

    return x


def lu(A):
    """
    Perform LU decomposition on a square matrix A.

    Arguments:
    A -- a square matrix

    Returns:
    L -- a lower triangular matrix
    U -- an upper triangular matrix

    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # TODO
    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    return L, U


def lu_solve(A, b):
    """
    Solve a linear system with a square matrix A using LU decomposition.

    Arguments:
    A -- a square matrix
    b -- a vector

    Returns:
    x -- the solution to the linear system

    """

    x = np.zeros(len(b))

    # TODO
    L, U = lu(A)
    y = solveLowerTriangular(L, b)
    x = solveUpperTriangular(U, y)

    return x
