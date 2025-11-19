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
from scipy.signal import butter, filtfilt, cheby1


def ex1(B=1):
   t = np.linspace(-3, 3, 1000)
   x = B * np.sinc(B * t) * np.sinc(B * t)

   Fs_values = [1.0, 1.5, 2.0, 4.0]

   fig, axs = plt.subplots(2, 2, figsize=(10, 8))
   axs = axs.flatten()

   i = 0
   for  Fs in Fs_values:
      Ts = 1 / Fs
      ax = axs[i]

      ax.plot(t, x, color='gray', linewidth=2, zorder=1)

      n_start = int(np.floor(-3 / Ts))
      n_end = int(np.ceil(3 / Ts))
      n_indices = np.arange(n_start, n_end + 1)

      n_Ts = n_indices * Ts

      x_n = B * np.sinc(B * n_Ts) * np.sinc(B * n_Ts)

      ax.stem(n_Ts, x_n)

      x1 = np.zeros_like(t)
      for n_val, val in zip(n_Ts, x_n):
         x1 += val * np.sinc((t - n_val) / Ts)

      ax.plot(t, x1, color='green', linestyle='--')

      ax.set_title(f'$F_s = {Fs:.2f}$ Hz')
      ax.set_xlabel('$t$ [s]')
      ax.set_ylabel('Amplitude')
      ax.grid(True, which='both', linestyle='-', linewidth=0.1, color='#e0e0e0')
      ax.set_xlim([-3, 3])
      ax.set_ylim([-0.2, 1.2])
      i += 1

   plt.tight_layout()
   plt.show()

def ex2():
   N=100
   x= np.random.rand(N)
   fig, axs = plt.subplots(2, 2, figsize=(10, 8))
   axs = axs.flatten()
   x_conv = x
   for i in range(4):
      axs[i].plot(x_conv)
      x_conv= np.convolve(x_conv,x)

  # plt.show()
   x1 = np.zeros(N)
   x1[10:20] = 1
   x_conv = x1
   for i in range(4):
      axs[i].plot(x_conv)
      x_conv= np.convolve(x_conv,x1)
   plt.show()

def ex3(N=10):
   p = np.random.randint(-50,50,N)
   q = np.random.randint(-50,50,N)
   r1 = np.convolve(p,q)

   L = 2 * N + 1
   fft_size = 2 ** (int(np.ceil(np.log2(L))))
   p_fft = np.fft.fft(p,fft_size)
   q_fft = np.fft.fft(q,fft_size)
   r2 = p_fft*q_fft
   r2_coef = (np.fft.ifft(r2))
   r2 = np.round(r2_coef.real).astype(int)
   print(r1)
   print(r2)

def ex4(N=20):
   x = np.random.rand(N)
   x[6:12] += 1000
   d = 9
   y = np.roll(x,d)
   X = np.fft.fft(x)
   Y = np.fft.fft(y)
   z1 = np.fft.ifft(np.conj(X)*Y).real
   z2 = np.fft.ifft(Y/(X)).real

   fig, axs = plt.subplots(2, 1, figsize=(10, 8))
   axs[0].plot(z1)
   axs[1].plot(z2)
   plt.show()


def fereastra_dreptunghiulara(Nw):
   return np.ones(Nw)


def fereastra_hanning(Nw):
   n = np.arange(Nw)
   return 0.5 * (1 - np.cos(2 * np.pi * n / (Nw - 1)))


def fft(signal, fs, Nw):
   Y = np.fft.fft(signal)
   Y_abs = np.abs(Y / Nw)
   Y_pozitiv = Y_abs[:Nw // 2 + 1]
   Y_pozitiv[1:-1] = 2 * Y_pozitiv[1:-1]

   f_axis = fs * np.arange(Nw // 2 + 1) / Nw
   return f_axis, Y_pozitiv

def ex5():
   Nw = 200
   fs = 2000
   f_sin = 100
   t = np.arange(Nw) / fs
   x = np.sin(2 * np.pi * f_sin * t)
   w_rect = fereastra_dreptunghiulara(Nw)
   w_hann = fereastra_hanning(Nw)
   x_rect = x * w_rect
   x_hann = x * w_hann
   f_rect, X_rect = fft(x_rect, fs, Nw)
   f_hann, X_hann = fft(x_hann, fs, Nw)

   fig, axs = plt.subplots(4, 1, figsize=(12, 8))
   axs[0].plot(t*1000, x_rect)
   axs[1].plot(f_rect,X_rect)
   axs[2].plot(t*1000, x_hann)
   axs[3].plot(f_hann,X_hann)
   plt.show()

def ButVsCheb(x,N,rp):
   fs = 1/3600
   fc = 1 / 8  # trec semnale mai lente de 8 ore\
   FNy = fs / 2 * 3600
   wn = fc / FNy
   bButter, aButter = butter(N=N, Wn=wn, btype='low', analog=False)

   yButter = filtfilt(bButter, aButter, x)
   bChebby, aChebby = cheby1(N=N, rp=rp, Wn=wn, btype='low', analog=False)
   yChebby = filtfilt(bChebby, aChebby, x)
   plt.plot(x, label='Semnal raw')
   plt.plot(yChebby, label='Filtru Chebby')
   plt.plot(yButter, label='Filtru Butterworth')
   plt.title(f'N={N} and rp={rp}')
   plt.legend()
   plt.show()
def ex6():
   # a si b-ul
   fs = 1 / 3600
   x = numpy.genfromtxt('lab6_data/Train.csv', delimiter=',', skip_header=1, usecols=2)
   x = x[0:72]
   plt.plot(x,label = 'semnal original')
   for w in [5,9,13,17]:
      y = (np.convolve(x,np.ones(w),'valid')/w)
      plt.plot(y, label=f'w={w}')
   plt.legend()
   plt.show()


   #F Nyquist = fs/2 => ciclu de 2 ore
   ButVsCheb(x,5,5)
   ButVsCheb(x,3,10)
   ButVsCheb(x,7,10)

ex5()