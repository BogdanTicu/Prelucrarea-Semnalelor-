import numpy as np
import matplotlib.pyplot as plt
from numpy import random

def Unu_c():

    # 6 = nr de esantioane cu frecventa 200Hz : 200*0.03 = 6
    x1 = np.linspace(0,.03,6)
    y1 = np.cos(520*np.pi*x1 + np.pi/3)

    x2 = np.linspace(0,0.03,6)
    y2 = np.cos(280*np.pi*x2 - np.pi/3)

    x3 = np.linspace(0,0.03,6)
    y3 = np.cos(120*np.pi*x3 + np.pi/3)


    n = 3

    fig, axs = plt.subplots(n)
    fig.suptitle('Titlu Principal')

    axs[0].stem(x1,y1)
    axs[1].stem(x2,y2)
    axs[2].stem(x3,y3)

    plt.show()

def Unu_b():

    nums = 0.03/0.0005
    # 6 = nr de esantioane cu frecventa 200Hz : 200*0.03 = 6
    x1 = np.linspace(0,.03,nums)
    y1 = np.cos(520*np.pi*x1 + np.pi/3)

    x2 = np.linspace(0,0.03,nums)
    y2 = np.cos(280*np.pi*x2 - np.pi/3)

    x3 = np.linspace(0,0.03,nums)
    y3 = np.cos(120*np.pi*x3 + np.pi/3)


    n = 3

    fig, axs = plt.subplots(n)
    fig.suptitle('Titlu Principal')

    axs[0].stem(x1,y1)
    axs[1].stem(x2,y2)
    axs[2].stem(x3,y3)

    plt.show()


def Doi():


    x0 = np.linspace(0, 1, 1600)
    y0 = np.sin(800 * np.pi * x0)
    plt.figure()
    plt.title('Exercitiul 2a')
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.plot(x0, y0)
    plt.stem(x0, y0)
    plt.show()

    x1 = np.linspace(0, 3)
    y1 = np.sin(1600 * np.pi * x1)
    plt.figure()
    plt.title('Exercitiul 2b')
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.plot(x1, y1)
    plt.stem(x1, y1)
    plt.show()

    x2 = np.linspace(0, 0.01, 240)
    y2 = np.mod(240 * x2, 1)
    plt.figure()
    plt.title('Exercitiul 2c')
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.plot(x2, y2)
    # plt.stem(x2, y2)
    plt.show()


    x3 = np.linspace(0,4, 300)
    y3 = np.sign(np.sin(2 * np.pi * 300 *x3))

    plt.figure()
    plt.title('Exercitiul 2d')
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.plot(x3, y3)
    plt.show()



    A = np.array(random.rand(128,128))

    plt.title('Exercitiul 2e')
    plt.imshow(A)
    plt.show()


    B = np.array(np.zeros((128,128)))
    plt.title('Exercitiul 2f')
    plt.imshow(B)
    plt.show()

Doi()

#La ex3 la a) intervalul de timp este 1/2000
#1 esantion = 4 biti -> 2 esantionare = 1 Byte -> 2000 esantioane intr-o secunda = 1000B -> 2000*3600 esantioane pe ora = 3600000 B
#
