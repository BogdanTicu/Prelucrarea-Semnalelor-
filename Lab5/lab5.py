import os

import numpy
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import sounddevice as sd
import math
import time

'''
    a)
    N=18288 esantioane in total. 
    s-a facut 1 esantion pe ora  -> frecventa = 1/3600 Hz = 0.0002777777777777778 Hz
    
    
    b)
    1 esantion/ora = 24 esantioane/zi. 
    18288 esantioane in 18288/24 = 762 zile.
    
    c) 
    frecventa maxima e fs/2 = 0.0001388888888888889
    
    e) semnalul este discret 
    
    f)
Pozitia 1:
  Val: 66.8539
  Frecventa: 0.0000000152 Hz
  Perioada (aprox.): 761.92 zile
Pozitia 2:
  Val: 35.2192
  Frecventa: 0.0000000304 Hz
  Perioada (aprox.): 380.96 zile
Pozitia 3:
  Val: 27.1020
  Frecventa: 0.0000115753 Hz
  Perioada (aprox.): 1.00 zile
Pozitia 4:
  Val: 25.2199
  Frecventa: 0.0000000456 Hz
  Perioada (aprox.): 253.97 zile
  
  Observam ca pozitia 1 se repeta cam la 2 ani, a2a dupa 1 an, a3a prezinta perioada de 1 zi, si a4a perioada de 8 luni
  
  h) Putem determina un ciclu zilnic/saptamanal, si dupa numaram cate cicluri de genul acesta sunt. 
  Dupa stim data de sfarsit a masuratorii numaram cate saptamani/zile avem de la data de sfarsit in spate si putem determina 
  data de inceput a masuratorii
'''

def d():
    fs = 1/3600
    x = numpy.genfromtxt('lab5_data/Train.csv',delimiter=',',skip_header=1,usecols=2)
    x = x - np.mean(x)
    N = len(x)
    Nhalf = N//2
    X = np.fft.fft(x)
    print(X[0])
    Xmod = abs(X/N)
    X_half = Xmod[:Nhalf]
    f = fs*np.linspace(0,Nhalf,Nhalf)/N

    plt.plot(f,X_half)
    plt.yscale('log')
    plt.show()

def f(max_values = 4):
    fs = 1 / 3600
    x = numpy.genfromtxt('lab5_data/Train.csv', delimiter=',', skip_header=1, usecols=2)
    x = x - np.mean(x)
    N = len(x)
    Nhalf = N // 2
    X = np.fft.fft(x)
    print(X[0])
    Xmod = abs(X / N)
    X_half = Xmod[:Nhalf]

    indices_max = np.argsort(X_half)[-max_values:][::-1]

    for i in range(max_values):
        idx = indices_max[i]
        frecventa_hz = f[idx]
        val = X_half[idx]

        perioada_ore = 1 / (frecventa_hz / fs)
        nr_zile = perioada_ore / 24

        print(f"Pozitia {i + 1}:")
        print(f"  Val: {val:.4f}")
        print(f"  Frecventa: {frecventa_hz:.10f} Hz")
        print(f"  Perioada (aprox.): {nr_zile:.2f} zile")
def g():
    fs = 1 / 3600
    x = numpy.genfromtxt('lab5_data/Train.csv', delimiter=',', skip_header=1, usecols=2)
    x = x - np.mean(x)
    N = len(x)
    Nhalf = N // 2
    X = np.fft.fft(x)
    Xmod = abs(X / N)
    X_half = Xmod[:Nhalf]
    f = fs * np.linspace(0, Nhalf, Nhalf) / N

    nr_zile = 14
    #8 oct 2012 - esantionul 1056 (de la 25 august pana la 8 oct sunt 44 de zile, deci 44*24=1056 esantioane).
    f1 = f[1056:1056+nr_zile*24]
    X_half = X_half[1056:1056+nr_zile*24]
    plt.plot(f1, X_half)
    plt.yscale('log')
    plt.show()

def i():
    fs = 1 / 3600
    x = numpy.genfromtxt('lab5_data/Train.csv', delimiter=',', skip_header=1, usecols=2)
    x = x - np.mean(x)
    N = len(x)
    Nhalf = N // 2
    X = np.fft.fft(x)
    Xmod = abs(X / N)
    X_half = Xmod[:Nhalf]
    f = fs * np.linspace(0, Nhalf, Nhalf) / N

    f_taiat = 1 / (24 * 3600)
    prag = int(f_taiat * N / fs)
    X[prag::N//2+1] = 0 #luam frecventele de peste prag si le anulam.
    X[N-prag::N] = 0 #luam frecventele opus conjugate
    x_nou = np.fft.ifft(X).real
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(x[1000:2000])
    ax[1].plot(x_nou[1000:2000])
    plt.show()
    
i()