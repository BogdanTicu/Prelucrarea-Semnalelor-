import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import sounddevice as sd
import math

import numpy as np

def FourierMatrix(N=64):
    F = np.zeros((N, N), dtype=complex)

    for k in range(N):
        for n in range(N):
            exponent = -1j * 2 * np.pi *( k * n)/ N
            F[k, n] = np.exp(exponent)
    return F

def checkUnitar(F,N):
    F_H = F.conj().T
    produs = F_H @ F
    matrice_identitate_scalata = N * np.eye(N)
    ok = np.allclose(produs, matrice_identitate_scalata)
    return ok
def Ex1():
    N = 64
    F = FourierMatrix(N)
    print(F)
    fig, ax = plt.subplots(5,1, figsize=(16, 24))
    plt.suptitle(f"Liniile Matricei Fourier (N={N}) ", fontsize=14)

    ax[0].plot(F[0].real)
    ax[0].plot(F[0].imag)
    ct = 1
    for k in [1,2]:
        ax[ct].plot(F[k].real)
        ax[ct].plot(F[k].imag)
        ct+=1
        ax[ct].plot(F[N-k].real)
        ax[ct].plot(F[N-k].imag)
        ct+=1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print(checkUnitar(F))

def Ex2():
    esantioane = 1000
    t = np.linspace(0,1,esantioane)
    f0 = 5
    x = np.sin(2*np.pi*f0*t)
    exponent = -1j * 2 * np.pi * t
    y = x * np.exp(exponent)
    plt.plot(y.real, y.imag)
    plt.show()

    fig, ax = plt.subplots(4, 1, figsize=(16, 24))
    ct = 0
    for w in [1,2,5,7]:
        exponent = -1j * 2 * np.pi * t * w
        y = x * np.exp(exponent)
        ax[ct].plot(y.real,y.imag)
        ct+=1
    plt.show()

def Ex3():
    N = 500
    t = np.linspace(0, 1, N, endpoint=False)

    f1 = 100
    f2 = f1 / 2
    f3 = f1 / 5


    x = np.sin(2 * np.pi * f1 * t + np.pi / 4) +  np.sin(2 * np.pi * f2 * t + np.pi / 6) + np.sin(2 * np.pi * f3  * t + np.pi / 2)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        n_indices = np.arange(N)
        exponent = -1j * 2 * np.pi * k * n_indices / N

        X[k] = np.sum(x * np.exp(exponent))
    X_abs = np.abs(X)
    N_half = N // 2
    Fs = 1000
    f_half = np.arange(N_half) * Fs / N
    X_half = X_abs[0:N_half]
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(t, x)
    ax[0].grid(True)

    ax[1].stem(f_half, X_half)
    ax[1].set_xlabel('Frecventa (Hz)')
    ax[1].grid(True)


    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


# Apelarea funcției:
# Ex3()
Ex3()