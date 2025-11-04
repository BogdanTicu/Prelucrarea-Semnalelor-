import os

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import sounddevice as sd
import math
import time
def FourierMatrix(N):
    F = np.zeros((N, N), dtype=complex)

    for k in range(N):
        for n in range(N):
            exponent = -1j * 2 * np.pi *( k * n)/ N
            F[k, n] = np.exp(exponent)
    return F

def fft(x, N):
    if N <= 1:
        return x

    X_even = fft(x[::2],N//2)
    X_odd = fft(x[1::2],N//2)
    m = np.arange(N//2)
    exp = (-1j*2*np.pi*m/N)
    X = np.concatenate((X_even + exp * X_odd, X_even - exp*X_odd))
    return X

def Ex1():
    Nvals = [128, 256, 512, 1024, 2048, 4096, 8192]
    time_dft = []
    time_fft = []
    time_fft_py = []

    for N in Nvals:
        x = np.random.rand(N)
        start_time = time.time()
        F = FourierMatrix(N)
        X = np.dot(F,x)
        end_time = time.time()
        time_dft.append(end_time-start_time)
        start_time = time.time()
        fft(x,N)
        end_time = time.time()
        time_fft.append(end_time - start_time)
        start_time = time.perf_counter()
        np.fft.fft(x)
        end_time = time.perf_counter()
        time_fft_py.append(end_time-start_time)

    '''time_dft = [0.07204365730285645, 0.1561441421508789, 0.7565503120422363, 2.980792999267578, 8.876270055770874,
     36.67830514907837, 141.5829153060913]
    time_fft = [0.004113674163818359, 0.005196809768676758, 0.014904022216796875, 0.01684713363647461, 0.0301516056060791,
     0.06188821792602539, 0.12364006042480469]'''
    print(time_dft)
    print(time_fft)
    print(time_fft_py)
    plt.yscale('log')
    plt.plot(Nvals,time_dft)
    plt.plot(Nvals, time_fft)
    plt.plot(Nvals, time_fft_py)
    plt.show()


def Ex2():
    t = np.linspace(0, 0.03, 1000)
    f0 = 100
    fs = 150
    Ts = 1 / fs
    tmodel = np.arange(0, 0.03, Ts)
    x1 = np.sin(2 * np.pi * f0 * tmodel)

    fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True)

    for k in range(0,4):
        f = f0+k*fs
        x = np.sin(2*np.pi*f*t)
        axs[k].plot(tmodel, x1, 'o', color = 'yellow', markeredgecolor='black', markersize=6)
        axs[k].plot(t,x,color='blue')
    plt.show()

def Ex3():
    t = np.linspace(0, 0.03, 1000)
    f0 = 100
    fs = 300
    Ts = 1 / fs
    tmodel = np.arange(0, 0.03, Ts)
    x1 = np.sin(2 * np.pi * f0 * tmodel)

    fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True)
    axs[0].plot(tmodel,x1, color='blue')
    axs[0].plot(tmodel, x1, 'o', color = 'yellow', markeredgecolor='black', markersize=6)
    ct = 1
    #l = [50, 250,125]
    for k in range(0,3):
        f = f0+k*fs
        x = np.sin(2*np.pi*f*t)
        axs[ct].plot(tmodel, x1, 'o', color = 'yellow', markeredgecolor='black', markersize=6)
        axs[ct].plot(t,x,color='blue')
        ct+=1
    plt.show()

#ex4: frecventa ar trb sa fie de 320Hz (200-40) * 2.

def Ex6():
    FS, data = wavfile.read('vocale.wav')
    if data.ndim > 1:
        data = data[:, 0]
    x = data.astype(np.float64)
    N_total = len(x)

    percent_window = 0.01
    desired_window_len = int(N_total * percent_window)

    window_len = 2 ** int(np.log2(desired_window_len))

    overlap = window_len // 2
    spectrogram_columns = []
    i = 0
    while i + window_len <= N_total:
        window = x[i: i + window_len]

        fft_result = fft(window,window_len)
        magnitude = np.abs(fft_result[:window_len // 2])

        spectrogram_columns.append(magnitude)
        i += (window_len - overlap)

    spectrogram_matrix = np.array(spectrogram_columns).T

    print(f" {window_len} eșantioane.")
    print(spectrogram_matrix.shape)


    plt.figure(figsize=(12, 6))
    max_freq = FS / 2
    freq_axis = np.linspace(0, max_freq, spectrogram_matrix.shape[0], endpoint=False)
    total_time = N_total / FS
    time_axis = np.linspace(0, total_time, spectrogram_matrix.shape[1], endpoint=False)


    magnitude_db = 20 * np.log10(spectrogram_matrix + 1e-10)

    vmax_val = np.max(magnitude_db)
    vmin_val = vmax_val - 100

    plt.pcolormesh(time_axis, freq_axis, magnitude_db,
                   shading='gouraud', cmap='inferno', vmin=vmin_val)

    plt.colorbar(label='Magnitudine (dB)')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Freq', fontsize=12)
    plt.ylim(0, max_freq / 2)
    plt.tight_layout()
    plt.show()

Ex6()