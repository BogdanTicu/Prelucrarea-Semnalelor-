import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def generate_semnal():
    N=1000
    t = np.linspace(0, 1, N)
    trend = t**2 + 0.02
    seasonal = 0.1 * np.sin(2 * np.pi * 5 * t)
    noise = 0.05 * np.random.normal(size=N)
    signal = trend + seasonal + noise
    return signal

def ex2():
    signal = generate_semnal()
    N = len(signal)
    alpha = 0.2
    signal_smooth = np.zeros(N)
    signal_smooth[0] = signal[0]

    for t in range(1, N):
        signal_smooth[t] = alpha * signal[t] + (1 - alpha) * signal_smooth[t - 1]

    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Semnal Original')
    plt.plot(signal_smooth, label='Semnal Neted', color='orange')
    plt.legend()
    plt.show()

    alpha_d = 0.2
    beta_d = 0.1
    m_d = 20
    # Variabilele de stare
    s_dubla = np.zeros(N)  # Nivelul (level)
    b_dubla = np.zeros(N)  # Trendul (trend)

    # Initializare
    s_dubla[0] = signal[0]
    b_dubla[0] = signal[1] - signal[0] if N > 1 else 0

    # Calcul
    for t in range(1, N):
        s_dubla[t] = alpha_d * signal[t] + (1 - alpha_d) * (s_dubla[t - 1] + b_dubla[t - 1])
        b_dubla[t] = beta_d * (s_dubla[t] - s_dubla[t - 1]) + (1 - beta_d) * b_dubla[t - 1]

    # Estimare
    estimare_dubla = s_dubla[-1] + np.arange(1, m_d + 1) * b_dubla[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Semnal Original')
    plt.plot(s_dubla, label='Semnal Neted dublu', color='orange')
    plt.legend()
    plt.show()
    alpha_t = 0.3
    beta_t = 0.1
    gamma_t = 0.6
    L_t = 12  # Lungimea sezonului
    m_t = 12
    N_t = len(signal)

    # Variabilele de stare
    s_tripla = np.zeros(N_t)  # Nivelul
    b_tripla = np.zeros(N_t)  # Trendul
    c_tripla = np.zeros(N_t)  # Sezonalitatea


    s_tripla[0] = signal[0]

    b_init = (signal[L_t] - signal[0]) / L_t if N_t > L_t else 0
    b_tripla[0] = b_init

    medie_primul_ciclu = np.mean(signal[:L_t])
    c_tripla[:L_t] = signal[:L_t] - medie_primul_ciclu


    for t in range(1, N_t):
        season_lag_index = t - L_t
        season_lag = c_tripla[season_lag_index] if season_lag_index >= 0 else c_tripla[t]

        s_tripla[t] = alpha_t * (signal[t] - season_lag) + (1 - alpha_t) * (s_tripla[t - 1] + b_tripla[t - 1])
        b_tripla[t] = beta_t * (s_tripla[t] - s_tripla[t - 1]) + (1 - beta_t) * b_tripla[t - 1]
        c_tripla[t] = gamma_t * (signal[t] - s_tripla[t] - b_tripla[t - 1]) + (1 - gamma_t) * season_lag


    estimare_tripla = np.zeros(m_t)
    for m_step in range(1, m_t + 1):
        c_final_cycle_idx = N_t - L_t + (m_step - 1) % L_t
        c_m_final = c_tripla[c_final_cycle_idx if c_final_cycle_idx < N_t else c_final_cycle_idx - L_t]

        estimare_tripla[m_step - 1] = s_tripla[-1] + m_step * b_tripla[-1] + c_m_final

    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Semnal Original')
    plt.plot(s_tripla, label='Semnal Neted triplu', color='orange')
    plt.legend()
    plt.show()


def MA(serie, q):
    predictii = []
    erori = []
    indici = []

    for i in range(q, len(serie)):
        window = serie[i - q: i]

        pred = np.mean(window)
        real = serie[i]

        eroare = real - pred

        predictii.append(pred)
        erori.append(eroare)
        indici.append(i)

    return np.array(predictii), np.array(erori), np.array(indici)
def ex3():
    semnal = generate_semnal()
    N = len(semnal)
    q = 5

    ma_calculata, epsilon_partial, indexi = MA(semnal, q)


    epsilon = np.zeros(N)
    epsilon[indexi] = epsilon_partial


    teta = np.array([0.5, 0.3, -0.2, 0.1, 0.05])

    if len(teta) != q:
        raise ValueError(f"Q trebuie sa fie q={q}")

    # Semnalul modelat MA(q)
    ma_modelat = np.zeros(N)
    ma_modelat[indexi] = ma_calculata

    for t in range(q, N):
        erori_trecute = epsilon[t - q:t][::-1]

        termen_eroare = np.dot(teta, erori_trecute)

        ma_modelat[t] = ma_calculata[t - q] + termen_eroare

    plt.figure(figsize=(10, 6))
    plt.plot(semnal, label='Semnal Original', alpha=0.7)
    plt.plot(indexi, ma_calculata,
             label=f'q={q}',
             linestyle='--', color='green')
    plt.plot(ma_modelat,
             label=f'Semnal Modelat MA({q})',
             color='red')

    plt.title(f'MA({q})')
    plt.legend()
    plt.show()


def ex4():
    signal = generate_semnal()
    N = len(signal)


    max_p = 3
    max_q = 3

    best_aic = np.inf
    best_order = None

    d = 1

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = SARIMAX(signal, order=(p, d, q), trend='c', enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False)

                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
                    best_model_results = results
            except Exception:
                continue

    if best_order:
        print(f"Parametri optimi {max_p},{max_q}): {best_order}")
        print(f"AIC minim: {best_aic:.2f}")

        # Extragerea valorilor modelate
        fitted_values = best_model_results.fittedvalues

        plt.figure(figsize=(10, 6))
        plt.plot(signal, label='Semnal Original', alpha=0.7)
        plt.plot(fitted_values, label=f'Modelat ARIMA{best_order} (fitted)', color='red')
        plt.title(f'Model ARIMA{best_order} (p, d, q) cu AIC minim')
        plt.legend()
        plt.show()
    else:
        print("Nu am gasit model ARIMA valid.")
