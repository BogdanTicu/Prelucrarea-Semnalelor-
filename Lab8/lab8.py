import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def ex1a():
    N=1000
    t = np.linspace(0, 1, N)
    trend = t**2 + 0.02
    seasonal = 0.1 * np.sin(2 * np.pi * 5 * t)
    noise = 0.05 * np.random.normal(size=N)
    signal = trend + seasonal + noise

    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)
    plt.plot(t, trend, label='Semnal Original')
    plt.subplot(4, 1, 2)
    plt.plot(t, seasonal, label='Trend', linestyle='--')
    plt.subplot(4, 1, 3)
    plt.plot(t, noise, label='Sezonier', linestyle='--')
    plt.subplot(4, 1, 4)
    plt.plot(t, signal, label='Zgomot', linestyle='--')
    plt.legend()
    plt.show()

def autocorelation(x):
    n = len(x)
    N = np.arange(n,0,-1)
    varianta = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    ans = r / (varianta * N)
    return ans

def ex1b():
    N=1000
    t = np.linspace(0, 1, N)
    trend = t**2 + 0.02
    seasonal = 0.1 * np.sin(2 * np.pi * 5 * t)
    noise = 0.05 * np.random.normal(size=N)
    signal = trend + seasonal + noise
    autocorr = autocorelation(signal)
    plt.figure(figsize=(10, 4))
    plt.plot(autocorr)
    plt.title('Functia de Autocorelatie')
    plt.show()

def ex1c(steps=10):
    N = 1000
    t = np.linspace(0, 1, N)
    trend = t ** 2 + 0.02
    seasonal = 0.1 * np.sin(2 * np.pi * 5 * t)
    noise = 0.05 * np.random.normal(size=N)
    signal = trend + seasonal + noise

    p = 4

    X = np.zeros((N - p, p))
    for i in range(p):
        X[:, i] = signal[i: N - p + i]
    y = signal[p:N]

    a = np.linalg.lstsq(X, y, rcond=None)[0]

    print(f"Coeficienti AR estimati (p={p}): {a}")

    predicted_series = signal[N - p:].copy()

    for n in range(steps):
        past_values = predicted_series[n: n + p][::-1]
        next_value = np.dot(a, past_values)
        predicted_series = np.append(predicted_series, next_value)

    forecast = predicted_series[p:]

    time_original_end = t[N - p:]
    time_forecast = np.linspace(t[-1] + (t[1] - t[0]), t[-1] + steps * (t[1] - t[0]), steps)

    plt.figure(figsize=(12, 6))
    plt.title(f'Predictie AR({p}) pentru Urmatoarele {steps} Puncte')

    plt.plot(t[N - 100:], signal[N - 100:], 'b-', label='Semnal Original')

    plt.plot(time_original_end, signal[N - p:], 'bo', label=f'Ultimile {p} Puncte (Input AR)')

    plt.plot(time_forecast, forecast, 'r--', label=f'Previziune AR (Next {steps} Steps)')
    plt.xlabel('Timp')
    plt.ylabel('Valoare Semnal')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

    print("\n## Rezultatul Predictiei Viitoare ##")
    for i, val in enumerate(forecast):
        print(f"Pasul {i + 1} (Valoarea Predicta): {val:.5f}")


def ex1d():
    N = 1000
    t = np.linspace(0, 1, N)
    trend = t ** 2 + 0.02
    seasonal = 0.1 * np.sin(2 * np.pi * 5 * t)
    noise = 0.05 * np.random.normal(size=N)
    signal = trend + seasonal + noise
    N = len(signal)

    split_point = int(0.8 * N)

    signal_train = signal[:split_point]
    signal_test = signal[split_point:]
    N_test = len(signal_test)

    p_values = range(1, 21)

    m = 1

    best_p = 0
    min_mse = np.inf
    mse_results = {}

    print(f"Incepe optimizarea ordinului p. Orizont de predictie (m): {m}")
    print("-" * 40)


    for p in p_values:

        if p >= len(signal_train) / 2:
            print(f"Atentie: p={p} este prea mare pentru setul de antrenare. Oprire.")
            break


        X_train = np.zeros((len(signal_train) - p, p))
        for i in range(p):
            X_train[:, i] = signal_train[i: len(signal_train) - p + i]

        y_train = signal_train[p:]


        a = np.linalg.lstsq(X_train, y_train, rcond=None)[0]


        signal_pred = np.zeros(N_test)

        current_memory = signal_train[-p:].copy()

        for t in range(N_test):

            predicted_value = np.dot(a, current_memory[::-1])

            signal_pred[t] = predicted_value

            current_memory = np.roll(current_memory, -1)
            current_memory[-1] = signal_test[t]

        mse = np.mean((signal_test - signal_pred) ** 2)
        mse_results[p] = mse

        print(f"P= {p:2d} -> MSE: {mse:.8f}")

        if mse < min_mse:
            min_mse = mse
            best_p = p

    print("-" * 40)
    print(f"Cea mai buna performanta: Ordin AR p = {best_p}, cu MSE = {min_mse:.8f}")

    plt.figure(figsize=(10, 6))
    plt.plot(list(mse_results.keys()), list(mse_results.values()), marker='o', linestyle='-', color='blue')
    plt.axvline(x=best_p, color='red', linestyle='--', label=f'Cel mai bun p={best_p}')

    plt.title('Eroarea Patratica Medie (MSE) vs. Ordinul AR (p) - Orizont m=1')
    plt.xlabel('Ordinul Modelului AR (p)')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_p, min_mse

ex1d()