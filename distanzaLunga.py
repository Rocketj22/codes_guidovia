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
        x_value = abs(distance_values[1] - distance_values[0])
        err_x = (2 ** 0.5) * 0.2 / (24 ** (1 / 2))

        time_values = [float(line.strip()) for line in lines[4:] if line.strip().replace('.', '', 1).isdigit()]
        y_value = np.mean(time_values)  # Media dei tempi
        err_y = np.std(time_values, ddof=1)   # ddof=1 per la deviazione standard campionaria

        dataset.append([x_value, y_value, err_x, err_y])

    return dataset


def manipulate_data(dataset):
    dataset = np.array(dataset)

    x_values = dataset[:, 0]
    y_values = dataset[:, 1]
    err_x = dataset[:, 2]
    err_y = dataset[:, 3]

    # new values needed for the plot
    # x : distances - y: time
    mean_velocity = x_values / y_values
    mean_time = y_values / 2

    # new uncertainties for the new data
    err_X = err_y / 2
    err_y = mean_velocity * ((err_x / x_values) ** 2 + (err_y / y_values) ** 2) ** 0.5

    dataset[:, 0] = mean_time
    dataset[:, 1] = mean_velocity
    dataset[:, 2] = err_X
    dataset[:, 3] = err_y

    return dataset


def linear_model(B, x):
    """ Modello lineare y = B[0] * x + B[1] """
    return B[0] * x + B[1]


def fit_orthogonal(x, y, err_x, err_y):
    """ Fit ortogonale con incertezze su x e y """
    model = Model(linear_model)
    data = RealData(x, y, sx=err_x, sy=err_y)  # sx e sy sono gli errori su x e y
    odr = ODR(data, model, beta0=[1, 0])  # Stima iniziale [pendenza, intercetta]
    output = odr.run()

    slope, intercept = output.beta
    err_slope, err_intercept = np.sqrt(np.diag(output.cov_beta))  # Incertezze parametri
    return slope, intercept, err_slope, err_intercept, output


def plot_data(dataset):
    dataset = np.array(dataset)
    x_values = dataset[:, 0]
    y_values = dataset[:, 1]
    err_x = dataset[:, 2]
    err_y = dataset[:, 3]
    err_y *= (0.2097600667330427)**0.5

    # Fit ortogonale
    slope, intercept, err_slope, err_intercept, output = fit_orthogonal(x_values, y_values, err_x, err_y)

    # Calcolo di R² per valutare la bontà del fit
    y_fit = slope * x_values + intercept
    residuals = y_values - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Calcolo del chi quadro
    chi2 = np.sum(((y_values - y_fit) / err_y) ** 2)
    dof = len(x_values) - 2  # gradi di libertà
    chi2_red = chi2 / dof
    print(chi2, chi2_red)
    print(((y_values - y_fit) / err_y) ** 2)

    # Calcolo dell'errore a posteriori
    err_Y = err_y * (chi2_red)**0.5

    plt.figure(figsize=(8, 6))
    plt.errorbar(x_values, y_values, xerr=err_x, yerr=err_y, fmt='o', capsize=5, capthick=1)
    plt.plot(x_values, y_fit, color='orange', linewidth=1,
             label=r"Fit $y = Ax + B$"
                   f"\nA: {slope:.2f} ± {err_slope:.2f}\n"
                   f"B: {intercept:.2f} ± {err_intercept:.2f}\n"
                   f"R²: {r_squared:.3f}")

    plt.xlabel("tempo [s]")
    plt.ylabel("velocità media [cm/s]")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    folder_path = r"C:\Users\loren\Desktop\distanza lunga"
    file_list = glob.glob(folder_path + "/*.txt")
    dataset = get_data(folder_path, file_list)
    dataset = manipulate_data(dataset)
    dataset = np.delete(dataset, 9, axis=0)
    plot_data(dataset)