from numpy.fft import fftshift
from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.fft import fft2, ifft2, fftshift, ifftshift
def teorie():
    X = datasets.face(gray=True)
    plt.imshow(X, cmap=plt.cm.gray)
    plt.show()

    Y = np.fft.fft2(X)
    freq_db = 20*np.log10(abs(Y))

    plt.imshow(freq_db)
    plt.colorbar()

    rotate_angle = 45
    X45 = ndimage.rotate(X, rotate_angle)
    plt.imshow(X45, cmap=plt.cm.gray)
    plt.show()

    Y45 = np.fft.fft2(X45)
    plt.imshow(20*np.log10(abs(Y45)))
    plt.colorbar()
    plt.show()

def ex1a():
    n1 = np.arange(-32, 32)/64
    n2 = np.arange(-32, 32)/64
    N1, N2 = np.meshgrid(n1, n2)
    X = np.sin(2 * np.pi * N1 + 3 * np.pi * N2)
    plt.figure(figsize=(2, 1))
    plt.subplot(1, 2, 1)
    plt.imshow(X)
    plt.title('Function: sin(2πn1 + 3πn2)')

    Y = np.fft.fft2(X)
    freq_db = 20 * np.log10(abs(Y) + 1e-10)

    plt.subplot(1, 2, 2)
    plt.imshow(freq_db)
    plt.title('Spectrum')
    plt.colorbar()
    plt.show()

def ex1b():
    n1 = np.arange(-32, 32)/64
    n2 = np.arange(-32, 32)/64
    N1, N2 = np.meshgrid(n1, n2)
    X = np.sin(4 * np.pi * N1) + np.cos(6 * np.pi * N2)

    plt.figure(figsize=(2, 1))
    plt.subplot(1, 2, 1)
    plt.imshow(X)
    plt.title('Function: sin(4πn1) + cos(6πn2)')

    Y = np.fft.fft2(X)
    freq_db = 20 * np.log10(abs(Y) + 1e-10)

    plt.subplot(1, 2, 2)
    plt.imshow(freq_db)
    plt.title('Spectrum')
    plt.colorbar()
    plt.show()

def f(a,b,c,d):

    n1 = np.arange(-32, 32)
    n2 = np.arange(-32, 32)
    N1, N2 = np.meshgrid(n1, n2)
    X = np.zeros(N1.shape, dtype=complex)
    X[a, b] = 1
    X[c, d] = 1

    X_shifted = fftshift(X)
    X_abs = np.abs(X_shifted)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(X_abs, cmap=plt.cm.gray, extent=[-32, 31, 31, -32])

    plt.colorbar()

    Y = np.fft.ifft2(X)

    freq_db = np.real(Y)

    plt.subplot(1, 2, 2)
    plt.imshow(freq_db, cmap=plt.cm.gray)

    plt.colorbar()
    plt.tight_layout()
    plt.show()
def ex1c():
   f(0, 5, 0, 59)

def ex1d():
    f(5, 0, 59, 0)

def ex1e():
    f(5, 5, 59, 59)


def calcul_snr(semnal_original, semnal_afectat):
    norma_semnal = np.linalg.norm(semnal_original)
    norma_zgomot = np.linalg.norm(semnal_original - semnal_afectat)

    if norma_zgomot == 0:
        return float('inf')

    snr = 20 * np.log10(norma_semnal / norma_zgomot)
    return snr


def ex2(freq_cutoff=30):

    X = datasets.face(gray=True)
    Y = np.fft.fft2(X)

    freq_db = 20 * np.log10(np.abs(Y))

    Y_cutoff = Y.copy()
    Y_cutoff[freq_db < freq_cutoff] = 0

    X_cutoff = np.fft.ifft2(Y_cutoff)
    X_cutoff = np.real(X_cutoff)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(X, cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.imshow(X_cutoff, cmap=plt.cm.gray)

    plt.show()
    return X_cutoff

def ex3():

    X = datasets.face(gray=True)

    freq_cutoff = 70
    pixel_noise = 200
    noise = np.random.randint(-pixel_noise, pixel_noise+1, size=X.shape)
    X_noisy = X + noise
    X_denoised = ex2(freq_cutoff)
    fix, axs = plt.subplots(1, 3)
    axs[0].set_title("Raton Original")
    axs[0].imshow(X, cmap=plt.cm.gray)
    axs[1].set_title("Raton Zgomotos")
    axs[1].imshow(X_noisy, cmap=plt.cm.gray)
    axs[2].set_title("Raton Curatat")
    axs[2].imshow(X_denoised, cmap=plt.cm.gray)
    plt.show()

ex3()
