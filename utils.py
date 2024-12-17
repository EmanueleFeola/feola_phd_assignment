import numpy as np
import matplotlib.pyplot as plt


def load_sto(file_path):
    """
    :param file_path: path to .sto file
    :return: dictionary with column names as keys, column data as value
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the header end
    for i, line in enumerate(lines):
        if line.strip().lower() == 'endheader':
            header_end_index = i
            break

    # Extract column names
    headers = lines[header_end_index + 1].strip().split()

    # Read the data into a dictionary
    data = {header: [] for header in headers}

    # Start reading rows after the headers
    for line in lines[header_end_index + 2:]:
        if line.strip():  # Skip empty lines
            values = list(map(float, line.split()))
            for header, value in zip(headers, values):
                data[header].append(value)

    return data


def get_muscle_activation_sin(plot_flag=False):
    cycles = 10  # how many sine cycles
    length = np.pi * 2 * cycles
    resolution = 2000  # how many datapoints to generate
    points = np.arange(0, length, length / resolution)
    muscle_activation = np.sin(points)
    muscle_activation = muscle_activation / 2
    muscle_activation = muscle_activation + 0.5
    data_time = np.linspace(0, 60 * 10, resolution)

    if plot_flag:
        plt.plot(data_time, muscle_activation)
        plt.title("Muscle Activation: simulated sinusoidal signal")
        plt.grid(True, color="grey", linewidth="0.4", linestyle="-")
        plt.show()
    return [muscle_activation, data_time]


def neural_to_muscle_activation(emg_arr, shape_factor, plot_time_arr=None):
    """
    :param shape_factor: A
    :param emg_arr: array of processed emg values (u in paper). [0, 1].
    :return: muscle activation computed from emg values
    """
    A = shape_factor  # shape factor
    u = np.array(emg_arr)
    a = np.exp(A * u) - 1
    a = a / (np.exp(A) - 1)

    if plot_time_arr is not None:
        plt.plot(plot_time_arr, u, color='b', linestyle='-', label="emg input")
        plt.plot(plot_time_arr, a, color='g', linestyle='-', label="activation output")
        plt.title("Muscle Activation: computed from EMG signal")
        plt.grid(True, color="grey", linewidth="0.4", linestyle="-")
        plt.legend()
        plt.show()

    return a


def plot_muscle_activation_curves():
    emg_arr = np.linspace(0, 1, 100)
    A_arr = np.linspace(-0.1, -5, 20)  # shape factor
    a_arr = np.array([])  # muscle activation

    for A_idx, A_curr in enumerate(A_arr):
        for emg_idx, emg_curr in enumerate(emg_arr):
            a_out = neural_to_muscle_activation(emg_curr, A_curr)
            a_arr = np.append(a_arr, a_out)
        # other color patterns: plasma, inferno, cividis, Blues, coolwarm, seismic, twilight, tab10
        # plt.plot(emg_arr, a_arr, label=f"A={A_curr:.2f}")  # emg vs muscle activation
        plt.plot(emg_arr, a_arr, label=f"A={A_curr:.2f}", color=plt.cm.Blues(0.3 + (A_idx * len(emg_arr) + emg_idx) / (len(A_arr) * len(emg_arr))))  # emg vs muscle activation
        a_arr = np.array([])  # reset
    plt.title("Neural to muscle activation")
    plt.grid(True, color="grey", linewidth="0.4", linestyle="-")
    plt.xlabel("EMG")
    plt.ylabel("Muscle activation")
    plt.legend()
    plt.show()
