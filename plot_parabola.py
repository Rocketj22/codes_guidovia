import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, Data, RealData
import glob


# distance values in [cm]
# in principio i dati sono disposti con "tempo" sulle ascisse
# in principio i dati sono disposti con "velocita' media" sulle ordinate
# poi gli assi sono invertiti per avere maggiore errore relativo sulle ordinate


def get_data(folder_path, file_list):
    dataset = []

    for file_path in file_list:
        with open(file_path, "r") as file:
            lines = file.readlines()

        distance_values = [float(lines[i].strip()) for i in range(1, 3)]  # dati in riga 1, 2
        y_value = abs(distance_values[1] - distance_values[0])
        err_y = (2 ** 0.5) * 0.2 / (24 ** (1 / 2))

        time_values = [float(line.strip()) for line in lines[4:] if line.strip().replace('.', '', 1).isdigit()]
        x_value = np.mean(time_values)  # Media dei tempi
        err_x = np.std(time_values, ddof=1)/len(time_values)   # ddof=1 per la deviazione standard campionaria

        dataset.append([x_value, y_value, err_x, err_y])

    return dataset


def linear_model(B, x):
    """ Modello lineare y = B[0] * x**2 + B[1] * x + B[2] """
    return B[0] * x**2 + B[1] * x + B[2]


def fit_orthogonal(x, y, err_x, err_y):
    """ Fit ortogonale con incertezze su x e y """
    model = Model(linear_model)
    data = RealData(x, y, sx=err_x, sy=err_y)  # sx e sy sono gli errori su x e y
    odr = ODR(data, model, beta0=[1, 0, 0])  # Stima iniziale [pendenza, intercetta]
    output = odr.run()

    alpha, beta, gamma = output.beta
    err_alpha, err_beta, err_gamma = np.sqrt(np.diag(output.cov_beta))  # Incertezze parametri
    return alpha, beta, gamma, err_alpha, err_beta, err_gamma, output


def plot_data(dataset):
    dataset = np.array(dataset)
    x_values = dataset[:, 0]
    y_values = dataset[:, 1]
    err_x = dataset[:, 2]
    err_y = dataset[:, 3]

    # commenatre questo paragrafo se si analizza la distanza lunga
    #y_values = np.cumsum(y_values)
    #X_values = np.copy(x_values)
    #X_values[0] /= 2
    #for i in range(1, len(x_values)):
    #    X_values[i] = X_values[i - 1] + 0.5 * (x_values[i] + x_values[i - 1] )
    #x_values = X_values

    # Fit ortogonale
    alpha, beta, gamma, err_alpha, err_beta, err_gamma, output = fit_orthogonal(x_values, y_values, err_x, err_y)

    # Calcolo di R² per valutare la bontà del fit
    y_fit = alpha * x_values ** 2 + beta * x_values + gamma
    residuals = y_values - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Calcolo del chi quadro
    chi2 = np.sum(((y_values - y_fit) / err_y) ** 2)
    dof = len(x_values) - 2  # gradi di libertà
    chi2_red = chi2 / dof
    #print(((y_values - y_fit) / err_y) ** 2)
    print(chi2, chi2_red)

    # Calcolo dell'errore a posteriori e nuovo fit
    #print(err_y)
    err_Y = err_y * (chi2_red)**0.5
    #print(err_Y)
    alpha, beta, gamma, err_alpha, err_beta, err_gamma, output = fit_orthogonal(x_values, y_values, err_x, err_Y)
    # Calcolo di R² per valutare la bontà del fit
    y_fit = alpha * x_values ** 2 + beta * x_values + gamma
    residuals = y_values - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    # Calcolo del chi quadro
    chi2 = np.sum(((y_values - y_fit) / err_y) ** 2)
    dof = len(x_values) - 2  # gradi di libertà
    chi2_red = chi2 / dof
    print(((y_values - y_fit) / err_y) ** 2)
    print(chi2, chi2_red)

    plt.figure(figsize=(8, 6))
    plt.errorbar(x_values, y_values, xerr=err_x, yerr=err_Y, fmt='o', capsize=5, capthick=1)
    plt.plot(x_values, y_fit, color='orange', linewidth=1,
             label=r"Fit $y = Ax^2 + Bx + C$"
                   f"\nA: {alpha:.3f} ± {err_alpha:.3f}\n"
                   f"B: {beta:.2f} ± {err_beta:.2f}\n"
                   f"C: {gamma:.2f} ± {err_gamma:.2f}\n"
                   f"R²: {r_squared:.3f}")

    plt.xlabel("tempo [s]")
    plt.ylabel("spazio [cm]")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    folder_path = r"C:\Users\loren\Desktop\distanza_lunga"
    file_list = glob.glob(folder_path + "/*.txt")
    dataset = get_data(folder_path, file_list)
    #dataset = np.delete(dataset, 9, axis=0) #distanza corta
    #dataset = np.delete(dataset, 9, axis=0) #distanza lunga
    dataset = np.delete(dataset, 6, axis=0) #distanza_lunga
    plot_data(dataset)