import time
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray


def qrgivens(A: NDArray) -> tuple[NDArray, NDArray]:
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(np.float64)

    for j in range(n):
        for i in range(m - 1, j, -1):
            k = i - 1
            a, b = R[k, j], R[i, j]

            r = np.sqrt(a**2 + b**2)
            c, s = a / r, b / r

            R[k, :], R[i, :] = c * R[k, :] + s * R[i, :], -s * R[k, :] + c * R[i, :]
            Q[:, k], Q[:, i] = c * Q[:, k] + s * Q[:, i], -s * Q[:, k] + c * Q[:, i]

    return Q, R


def test_correctness():
    np.random.seed(42)
    for m, n in [(5, 3), (6, 6), (10, 8), (16, 12), (34, 34)]:
        A = np.random.randn(m, n)
        Q, R = qrgivens(A)

        assert np.allclose(Q @ R, A), "A is not equal QR !!!"
        assert np.allclose(Q @ Q.T, np.eye(Q.shape[0])), "Q is not orthogonal!!!"
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[0])), "Q is not orthogonal!!!"
        assert np.allclose(np.tril(R, -1), 0), "R is not upper triangular!!!"

    print("All tests passed!")


def test_performance(repeats=5):
    sizes = [50, 100, 200, 400, 600]
    times_custom_all = []
    times_numpy_all = []

    for size in sizes:
        times_custom = []
        times_numpy = []
        A = np.random.randn(size, size)

        for _ in range(repeats):
            start = time.time()
            qrgivens(A)
            times_custom.append(time.time() - start)

            start = time.time()
            np.linalg.qr(A)
            times_numpy.append(time.time() - start)

        times_custom_all.append(times_custom)
        times_numpy_all.append(times_numpy)

    means_custom = [np.mean(t) for t in times_custom_all]
    stds_custom = [np.std(t) for t in times_custom_all]
    means_numpy = [np.mean(t) for t in times_numpy_all]
    stds_numpy = [np.std(t) for t in times_numpy_all]

    plt.figure(figsize=(8, 5))
    plt.errorbar(sizes, means_custom, yerr=stds_custom, marker="o", capsize=5, label="Implementacja własna")
    plt.errorbar(sizes, means_numpy, yerr=stds_numpy, marker="s", capsize=5, label="NumPy np.linalg.qr")
    plt.xlabel("Rozmiar macierzy (n x n)")
    plt.ylabel("Czas wykonania [s]")
    plt.title("Porównanie czasu wykonania rozkładu QR (średnia ± std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_correctness()
    # test_performance()

    A = np.array([[12, -51, 4, 1], [6, 167, -68, 2], [-4, 24, -41, 3], [-1, 1, 0, 5]], dtype=np.float64)

    Q_custom, R_custom = qrgivens(A)
    Q_np, R_np = np.linalg.qr(A)

    print("Q (własna implementacja):\n", np.round(Q_custom, 4))
    print("R (własna implementacja):\n", np.round(R_custom, 4))
    print("\nQ (NumPy):\n", np.round(Q_np, 4))
    print("R (NumPy):\n", np.round(R_np, 4))

    print("\nNorma różnicy A - Q*R (własna):", np.linalg.norm(A - Q_custom @ R_custom))
    print("Norma różnicy A - Q*R (NumPy):", np.linalg.norm(A - Q_np @ R_np))
