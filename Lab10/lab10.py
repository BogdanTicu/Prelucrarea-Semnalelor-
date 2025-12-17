from lab9 import generate_semnal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def stationar(coeficienti):
    coeficienti = np.concatenate(([1], -coeficienti))
    radacini = radacini_polinom(coeficienti)
    moduluri = np.abs(radacini)
    if(np.all(moduluri > 1)):
        print("Da, este stationar")
    else:
        print("Nu este stationar")

def ex2(steps=10):
    signal = generate_semnal()
    N = len(signal)
    p = 4
    t = np.linspace(0, 1, N)

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
    stationar(a)
    plt.figure(figsize=(12, 6))
    plt.title(f'Predictie AR({p}) pentru Urmatoarele {steps} Puncte')

    plt.plot(t[N - 100:], signal[N - 100:], 'b-', label='Semnal Original')

    plt.plot(time_original_end, signal[N - p:], 'bo', label=f'Ultimele {p} Puncte (Input AR)')

    plt.plot(time_forecast, forecast, 'r--', label=f'AR next {steps} steps')
    plt.xlabel('Timp')
    plt.ylabel('Valoare Semnal')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

    for i, val in enumerate(forecast):
        print(f"Pasul {i + 1} (valoarea prezisa): {val:.5f}")

def ex3(steps=10):
    signal = generate_semnal()
    N = len(signal)
    p = 20
    t = np.linspace(0, 1, N)

    # =========================
    # Constructia AR(p)
    # =========================
    X = np.zeros((N - p, p))
    for i in range(p):
        X[:, i] = signal[i: N - p + i]

    y = signal[p:N]

    def greedy_ar(X, y, max_features):
        N, p = X.shape
        selected = []
        remaining = list(range(p))
        coef = np.zeros(p)

        for _ in range(max_features):
            best_err = np.inf
            best_j = None

            for j in remaining:
                idx = selected + [j]
                X_sub = X[:, idx]
                a_sub = np.linalg.lstsq(X_sub, y, rcond=None)[0]
                err = np.mean((y - X_sub @ a_sub) ** 2)

                if err < best_err:
                    best_err = err
                    best_j = j
                    best_coef = a_sub

            selected.append(best_j)
            remaining.remove(best_j)

        coef[selected] = best_coef
        return coef, selected

    a_greedy, selected_lags = greedy_ar(X, y, max_features=5)

    alpha = 0.01
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    model.fit(X, y)
    a_lasso = model.coef_

    def forecast_ar(signal, a, steps):
        p = len(a)
        predicted = signal[-p:].copy()
        for _ in range(steps):
            next_val = np.dot(a, predicted[-p:][::-1])
            predicted = np.append(predicted, next_val)
        return predicted[p:]

    forecast_greedy = forecast_ar(signal, a_greedy, steps)
    forecast_lasso = forecast_ar(signal, a_lasso, steps)

    time_forecast = np.linspace(
        t[-1] + (t[1] - t[0]),
        t[-1] + steps * (t[1] - t[0]),
        steps
    )
    stationar(a_greedy)
    stationar(a_lasso)
    plt.figure(figsize=(12, 6))
    plt.plot(t[-100:], signal[-100:], 'b-', label='Semnal original')
    plt.plot(time_forecast, forecast_greedy, 'r--', label='Predictie AR Greedy')
    plt.plot(time_forecast, forecast_lasso, 'g--', label='Predictie AR L1')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.title('Model AR Greedy vs L1')
    plt.show()

    print("Coeficienti AR Greedy:", a_greedy)
    print("Lag-uri selectate:", [i + 1 for i in selected_lags])

    print("Coeficienti L1:", a_lasso)
    print("Lag-uri selectate:", np.where(np.abs(a_lasso) > 1e-6)[0] + 1)

def radacini_polinom(coeficienti):
    coeficienti = np.array(coeficienti,dtype=float)
    coeficienti_normalizati = coeficienti[1:] / coeficienti[0]
    if len(coeficienti_normalizati) < 2: # polinom constant sau de gradul 0
        return np.array([])
    matriceCompanion = np.zeros((len(coeficienti_normalizati), len(coeficienti_normalizati)))
    for i in range(len(coeficienti_normalizati)-1):
        matriceCompanion[i+1][i] = 1
    matriceCompanion[:, -1] = -coeficienti_normalizati[::-1] # negam coeficientii de pe ultima coloana
    radacini = np.linalg.eigvals(matriceCompanion)
    return radacini
def ex4():
    coeficienti = [1,5,6]
    radacini = radacini_polinom(coeficienti)
    print(radacini)


ex2()