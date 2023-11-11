import numpy as np


def lu_factorization(A):
    if A[0, 0] == 0:
        return 'Not possible'
    n = len(A)
    L = np.zeros((n, n)) + np.eye(n)
    U = np.zeros((n, n))
    U[0, 0] = A[0, 0]

    for j in range(1, n):
        U[0, j] = A[0, j] / L[0, 0]
        L[j, 0] = A[j, 0] / U[0, 0]

    for i in range(1, n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.sum(L[i, :i] * U[:i, j])
        if U[i, i] == 0:
            return 'Not possible'
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * U[:i, i])) / U[i, i]

    return L, U


A = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])

print(lu_factorization(A))
