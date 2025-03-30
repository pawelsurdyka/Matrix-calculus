import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from timeit import timeit
from numpy.typing import NDArray


def backward_substitution(A: NDArray) -> NDArray:
    """
    The input matrix `A` is the augmented, upper-triangular matrix of the system of equations.
    """
    n = A.shape[0]
    x = np.empty(n)

    x[n - 1] = A[n - 1, -1] / A[n - 1, n - 1]
    for i in reversed(range(n - 1)):
        x[i] = (A[i, -1] - np.sum(A[i, i + 1 : -1] * x[i + 1 :])) / A[i, i]

    return x


def ge_simple(A: NDArray) -> NDArray:
    """
    The input matrix `A` is the augmented, matrix of the system of equations. This implementation
    does not use pivoting and can be unstable.
    """
    n = A.shape[0]
    A = A.copy().astype(np.float64)

    for i in range(n):
        A[i, :] /= A[i, i]
        for j in range(i + 1, n):
            A[j, i:] -= A[j, i] * A[i, i:]

    x = backward_substitution(A)
    return x


def ge_pivot(A: NDArray) -> NDArray:
    """
    The input matrix `A` is the augmented, matrix of the system of equations.
    """
    n = A.shape[0]
    A = A.copy().astype(np.float64)

    for i in range(n - 1):
        p = max(range(i, n), key=lambda p: abs(A[p, i]))
        A[[i, p], :] = A[[p, i], :]

        for j in range(i + 1, n):
            A[j, i:] -= A[j, i] / A[i, i] * A[i, i:]

    x = backward_substitution(A)
    return x


def lu_simple(A: NDArray) -> tuple[NDArray, NDArray]:
    """
    The input matrix `A` is any real-valued square matrix of shape (n,n).
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(np.float64)

    for i in range(n - 1):
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return L, U


def lu_pivot(A: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    The input matrix `A` is any real-valued square matrix of shape (n,n).
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(np.float64)
    P = np.eye(n)

    for i in range(n - 1):
        p = max(range(i, n), key=lambda p: abs(U[p, i]))

        U[[i, p], i:] = U[[p, i], i:]
        L[[i, p], :i] = L[[p, i], :i]
        P[[i, p], ::] = P[[p, i], ::]

        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return L, U, P


def test(plot=False):
    np.random.seed(42)

    print("--- GAUSSIAN ELIMINATION")
    print("Random testing on well-conditioned matrices")
    n = 34
    trials = 1_000
    for _ in trange(trials):
        A = np.random.rand(n, n + 1)
        x = ge_simple(A)
        assert np.allclose(A[:, :-1] @ x, A[:, -1])
        x = ge_pivot(A)
        assert np.allclose(A[:, :-1] @ x, A[:, -1])
    print("Done!")

    print("Test on ill-conditioned matrix")
    A = np.array([[1, -1, 1, 3], [2, -2, 4, 8], [3, 0, -9, 0]])
    x = ge_pivot(A)
    assert np.allclose(A[:, :-1] @ x, A[:, -1])
    print("Done!")

    if plot:
        print("Benchmark speed")
        repeats = 10
        sizes, times = [], {"naive": [], "pivot": []}
        for n in range(2, 11):
            print(f"n={n}")
            A = np.random.rand(2**n, 2**n + 1)
            x = ge_simple(A)
            assert np.allclose(A[:, :-1] @ x, A[:, -1])
            x = ge_pivot(A)
            assert np.allclose(A[:, :-1] @ x, A[:, -1])
            sizes.append(2**n)
            times["naive"].append(timeit(lambda: ge_simple(A), number=repeats))
            times["pivot"].append(timeit(lambda: ge_pivot(A), number=repeats))
        print("Done!")

        plt.scatter(sizes, times["naive"], color="None", edgecolors="k", label="GE simple")
        plt.scatter(sizes, times["pivot"], color="None", edgecolors="r", label="GE pivot")
        plt.ylabel(f"Execution time, best of {repeats} repeats [s]")
        plt.xlabel(r"Matrix size $n$")
        plt.legend()
        plt.show()

    print("\n--- LU FACTORIZATION")
    print("Random testing on well-conditioned matrices")
    n = 34
    trials = 1_000
    for _ in trange(trials):
        A = np.random.rand(n, n)
        L, U = lu_simple(A)
        assert np.allclose(L @ U, A)
        L, U, P = lu_pivot(A)
        assert np.allclose(L @ U, P @ A)
    print("Done!")

    print("Test on ill-conditioned matrix")
    A = np.array([[1, -1, 1], [2, -2, 4], [3, 0, -9]])
    L, U, P = lu_pivot(A)
    assert np.allclose(L @ U, P @ A)
    print("Done!")

    if plot:
        print("Benchmark speed")
        repeats = 10
        sizes, times = [], {"naive": [], "pivot": []}
        for n in range(2, 11):
            print(f"n={n}")
            A = np.random.rand(2**n, 2**n)
            L, U = lu_simple(A)
            assert np.allclose(L @ U, A)
            L, U, P = lu_pivot(A)
            assert np.allclose(L @ U, P @ A)
            sizes.append(2**n)
            times["naive"].append(timeit(lambda: lu_simple(A), number=repeats))
            times["pivot"].append(timeit(lambda: lu_pivot(A), number=repeats))
        print("Done!")

        plt.scatter(sizes, times["naive"], color="None", edgecolors="k", label="LU simple")
        plt.scatter(sizes, times["pivot"], color="None", edgecolors="r", label="LU pivot")
        plt.ylabel(f"Execution time, best of {repeats} repeats [s]")
        plt.xlabel(r"Matrix size $n$")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    test(plot=True)
