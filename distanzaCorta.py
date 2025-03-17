import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
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
        err_y = np.std(time_values, ddof=1)  # ddof=1 per la deviazione standard campionaria

        dataset.append([x_value, y_value, err_x, err_y])

    return dataset


def manipulate_data(dataset):
    dataset = np.array(dataset)

    x_values = dataset[:, 0]
    y_values = dataset[:, 1]
    err_x = dataset[:, 2]
    err_y = dataset[:, 3]

   # print(dataset)

    # new values needed for the plot
    # x : distances - y: time
    mean_velocity = x_values / y_values
    time = [y_values[0] / 2]
    for i in range(1, len(y_values)):
        time.append(time[i - 1] + 0.5 * y_values[i - 1] + 0.5 * y_values[i])

    # new uncertainties for the new data
    # x - total time - y : mean_speed
    err_X = [err_y[0] / 2]
    for i in range(1, len(err_x)):
        err_X.append(((err_X[i-1])**2 + (0.5 * err_y[i-1])**2 + (0.5 * err_y[i])**2) **0.5)
    err_y = mean_velocity * ((err_x/x_values)**2 + (err_y/y_values)**2 )**0.5

    dataset[:, 0] = time
    dataset[:, 1] = mean_velocity
    dataset[:, 2] = err_X
    dataset[:, 3] = err_y

    return dataset


def friction(dataset):
    dataset = np.array(dataset)
    x_values = dataset[:, 0]
    y_values = dataset[:, 1] / 100
    err_x = dataset[:, 2]
    err_y = dataset[:, 3]/ 100

    # Definisci i parametri iniziali
    g = 9.8  # Accelerazione di gravità (m/s^2)
    alpha = 60 / 60 * np.pi / 180  # Inclinazione del piano (radianti)
    v0 = 0.2875  # Velocità iniziale (m/s)
    x0 = .4  # Posizione iniziale (m)
    t_max = 1.5  # Tempo massimo di simulazione (s)
    mu = 0.002  # Coefficiente di attrito radente dinamico
    b = 0.1  # Coefficiente di attrito viscoso

    # Calcola l'accelerazione lungo il piano inclinato (con attrito radente)
    a_no_friction = g * np.sin(alpha)
    a_with_kinetic_friction = g * (np.sin(alpha) - mu * np.cos(alpha))

    # Crea un array di tempi
    t = np.linspace(0, t_max, 1000)
    dt = t[1] - t[0]  # Intervallo di tempo

    # Calcola velocità, posizione e accelerazione in funzione del tempo senza attrito
    v_no_friction = v0 + a_no_friction * t
    x_no_friction = x0 + v0 * t + 0.5 * a_no_friction * t ** 2
    a_array_no_friction = np.full_like(t, a_no_friction)  # Accelerazione costante senza attrito

    # Calcola velocità, posizione e accelerazione in funzione del tempo con attrito radente
    v_with_kinetic_friction = v0 + a_with_kinetic_friction * t
    x_with_kinetic_friction = x0 + v0 * t + 0.5 * a_with_kinetic_friction * t ** 2
    a_array_with_kinetic_friction = np.full_like(t,
                                                 a_with_kinetic_friction)  # Accelerazione costante con attrito radente

    # Calcola velocità, posizione e accelerazione in funzione del tempo con attrito radente e viscoso
    a_with_viscous_friction = a_with_kinetic_friction - b * v_with_kinetic_friction
    v_with_viscous_friction = np.cumsum(a_with_viscous_friction * dt) + v0
    x_with_viscous_friction = np.cumsum(v_with_viscous_friction * dt) + x0
    a_array_with_viscous_friction = a_with_viscous_friction\


    # Funzione per il fitting lineare
    def linear_func(x, a, b):
        return a + b * x

    # Fitting dei dati v_bar
    v_bar_params, pcov = curve_fit(linear_func, x_values, y_values)
    v_bar_a, v_bar_b = v_bar_params

    # Crea il grafico
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

    # Grafico accelerazione
    plt.plot(t, a_array_no_friction, label='Senza attrito')
    plt.plot(t, a_array_with_kinetic_friction, '--', label='Con attrito radente')
    plt.plot(t, a_array_with_viscous_friction, '-.', label='Con attrito radente e viscoso')
    plt.xlabel('tempo (s)')
    plt.ylabel('accelerazione [m/s^2]')
    plt.title('Accelerazione lungo il piano inclinato')
    plt.legend()

    plt.show()

    # Grafico velocità
    plt.plot(t, v_no_friction, label='Senza attrito')
    plt.plot(t, v_with_kinetic_friction, '--',
             label=r'Con attrito radente')
    plt.plot(t, v_with_viscous_friction, '-.', label='Con attrito radente e viscoso')
    # ax2.scatter(table_times, table_velocities, c='r', marker='o', label='Dati dalla tabella')
    # ax2.errorbar(table_times, table_velocities, yerr=table_velocities_sigma, fmt='o', color='r', label='Dati')
    # ax2.plot(t_bar, [linear_func(x, v_bar_a, v_bar_b) for x in t_bar], 'g--', label=f'Fitting lineare (v_bar): a={v_bar_a:.4f}, b={v_bar_b:.4f}')
    plt.errorbar(x_values, y_values, yerr=err_y, fmt='o', color='r', capsize=2, capthick=1, label='Dati')
    plt.xlabel('tempo [s]')
    plt.ylabel('velocità [m/s]')
    plt.title('Velocità lungo il piano inclinato')
    plt.legend()

    #plt.tight_layout()
    plt.show()
    print(v_bar_params)

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
    err_y *= (0.10754230511945359)**0.5

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

    # Calcolo dell'errore a posteriori
    err_Y = err_y * (chi2_red) ** 0.5
    #print(err_Y, err_y)

    plt.figure(figsize=(8, 6))
    plt.errorbar(x_values, y_values, xerr=err_x, yerr=err_Y, fmt='o', capsize=5, capthick=1)
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
    folder_path = r"C:\Users\loren\Desktop\distanza_corta"
    file_list = glob.glob(folder_path + "/*.txt")
    dataset = get_data(folder_path, file_list)
    dataset = manipulate_data(dataset)
    # friction(dataset)
    #dataset = np.delete(dataset, 2, axis=0)
    #dataset = np.delete(dataset, 1, axis=0)
    plot_data(dataset)
