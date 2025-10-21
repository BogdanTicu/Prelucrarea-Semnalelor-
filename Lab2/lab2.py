import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import sounddevice as sd


def Ex1():
    #A = 2, f0 = 800
    x1 = np.linspace(0, 3)
    y1 = 2 * np.sin(1600 * np.pi * x1)
    fig, axs = plt.subplots(2)
    fig.suptitle('Ex1')

    x2 = np.linspace(0, 3)
    y2 = 2 * np.cos(1600 * np.pi * x1)
    axs[0].plot(x1, y1)
    axs[1].plot(x2, y2)
    plt.show()


def Ex2():
    #A = 1
    faza = [1, 2, 3, 4]
    '''fig, axs = plt.subplots(4)
        fig.suptitle('Ex2')
        for i in range(1, 5):
            x1 = np.linspace(0, 1)
            y1 = 1 * np.sin(2 * np.pi * 100 * x1 + faza[i - 1])
            axs[i - 1].plot(x1, y1)

        plt.show()
        '''
    fig, axs = plt.subplots(4)
    fig.suptitle('Ex2')
    x1 = np.linspace(0, 1,100)
    y1 = 1 * np.sin(2 * np.pi * 100 * x1 + faza[0])

    zgomot = np.random.normal(0,1,len(x1))
    SNR = [0.1, 1, 10, 100]
    ct = 0
    for snr in SNR:
        eta = np.sqrt((np.linalg.norm(x1)**2) / ((np.linalg.norm(zgomot)**2) * snr))
        x_nou = y1 + eta*zgomot
        axs[ct].plot(x_nou)
        ct += 1
    plt.show()
def Ex3():
    x0 = np.linspace(0, 3, 16000)
    y0 = np.sin(800 * np.pi * x0)

    x1 = np.linspace(0, 3, 10000)
    y1 = np.sin(1600 * np.pi * x1)

    x2 = np.linspace(0, 1, 24000)
    y2 = np.mod(240 * x2, 1)

    x3 = np.linspace(0, 3, 18000)
    y3 = np.sign(np.sin(600 * np.pi * x3))

    fs = 44100
    rate = int(10e5)
    files = ['a.wav','b.wav','c.wav','d.wav']
    for file in files:
        rate, x = wavfile.read(file)
        sd.play(x, fs)
        sd.wait()


def Ex4():
    x0 = np.linspace(0, 3, 1600)
    y0 = np.sin(800 * np.pi * x0)

    x2 = np.linspace(0, 1, 1600)
    y2 = np.mod(240 * x2, 1)

    y_final = y0 + y2
    fig, axs = plt.subplots(3)
    fig.suptitle('Ex4')
    axs[0].plot(y0)
    axs[1].plot(y2)
    axs[2].plot(y_final)
    plt.show()

def Ex5():
    x0 = np.linspace(0,1,10000)
    y0 = np.sin(400*np.pi*x0) #frecventa = 200
    y1 = np.sin(1000*np.pi*x0) #frecventa 500
    y = np.concatenate((y0,y1))
    rate = int(10e5)
    fs=44100
    wavfile.write('5.wav',rate,y)
    rate, x = wavfile.read('5.wav')

    sd.play(x,fs)
    sd.wait()
    #indiferent cate secunde setez la x0 al 2lea sunet ruleaza mult mai putin.

def Ex6():
    x0 = np.linspace(0,1,100)
    fs = 1000
    y1 = np.sin(2*np.pi*(fs/2)*x0)
    y2 = np.sin(2 * np.pi * (fs / 4) * x0)
    y3 = np.sin(2 * np.pi * 0 * x0)
    fig, axs = plt.subplots(3)
    fig.suptitle('Ex6')
    axs[0].plot(y1)
    axs[1].plot(y2)
    axs[2].plot(y3)
    plt.show()

def Ex7():
    fs = 1000
    x0 = np.linspace(0,1,fs)
    y0 = np.sin(2*np.pi*100*x0)

    y1 = []
    i = 0
    while i < len(y0):
        y1.append(y0[i])
        i+=4
    y1 = np.array(y1)


    y2 = []
    i = 1
    while i < len(y0):
        y2.append(y0[i])
        i+=4
    y2 = np.array(y2)
    fig, axs = plt.subplots(3)
    fig.suptitle('Ex7')
    axs[0].plot(y0)
    axs[1].plot(y1)
    axs[2].plot(y2)
    plt.show()

def Ex8():
    x = np.linspace(-np.pi/2, np.pi/2, 100)
    y1 = np.sin(x) #=functia sin
    y2 = x # functia liniara
    y_pade = (x - 7*(x**3/60))/(1+x**2/20)
    eroare_taylor = abs(y1-y2)
    eroare_pade = abs(y1-y_pade)
    fig, axs = plt.subplots(5)
    fig.suptitle('Ex8')
    plt.yscale('log')
    axs[0].plot(y1)
    axs[1].plot(y2)
    axs[2].plot(y_pade)
    axs[3].plot(eroare_taylor)
    axs[4].plot(eroare_pade)

    plt.show()




Ex8()