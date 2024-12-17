from utils import *
from Muscle import Muscle


def get_muscle_activation():
    ### define sinusoidal muscle activation
    [muscle_activation, data_time] = get_muscle_activation_sin(plot_flag=False)

    ### define muscle activation from EMG signal
    # filepath_emg = "D:\\Biomechpro_Lifting\\ExpMultiJoint\\Sub07\\OS\\DataFiles\\Sub07squat_15_2EMG.sto"
    # data_emg = load_sto(filepath_emg)
    # data_emg_bicep1 = data_emg["bicep1"]
    # data_time = data_emg["time"]
    # shape_factor = -0.1
    # # muscle_activation = neural_to_muscle_activation(emg_arr=data_emg_bicep1, shape_factor=shape_factor, plot_time_arr=None)
    # muscle_activation = neural_to_muscle_activation(emg_arr=data_emg_bicep1, shape_factor=shape_factor, plot_time_arr=data_time)
    # # plot_muscle_activation_curves() # check potvin non-linear transformation

    return [muscle_activation, data_time]


def exec_part_1():
    [muscle_activation, data_time] = get_muscle_activation()

    ### exec model
    m1 = Muscle()
    m1.set_activation(muscle_activation)
    mtu_len_arr = [0.30, 0.31, 0.32]

    for mtu_len_curr in mtu_len_arr:
        m1.set_fiber_len(mtu_len_curr)
        fiber_len = m1.get_fiber_len()
        print(f"fiber_len {fiber_len}")
        muscle_force = m1.compute_f_m(normed_flag=False, plot_time_vector=None)  # data_time
        plt.plot(data_time, muscle_force, label=f"{mtu_len_curr} m")

    # plt.title("Muscle Force with sine wave muscle activation input")
    plt.title("Muscle Force with muscle activation computed from EMG signal")
    plt.legend()
    plt.grid(True, color="grey", linewidth="0.4", linestyle="-")
    plt.xlabel("Time [s]")
    plt.ylabel("Force (N)")
    plt.show()


def exec_part_2():
    [muscle_activation, data_time] = get_muscle_activation()
    m1 = Muscle()
    m1.set_activation(muscle_activation)

    ### week 0: starting condition
    mtu_len = 0.31
    m1.set_fiber_len(mtu_len)
    muscle_force_w0 = m1.compute_f_m(normed_flag=False, plot_time_vector=None)
    plt.plot(data_time, muscle_force_w0, label="w0")
    # m1.plot_force_curves(hold_on_flag=True)
    print(m1)

    ### week n: simulate eccentric training
    m1.update_n_sarc_series(1.1)
    m1.update_n_sarc_parall(1.1)
    muscle_force_w1 = m1.compute_f_m(normed_flag=False, plot_time_vector=None)
    plt.plot(data_time, muscle_force_w1, label="w1")
    # m1.plot_force_curves()
    print(m1)

    plt.title("after n week of eccentric training")
    plt.legend()
    plt.grid(True, color="grey", linewidth="0.4", linestyle="-")
    plt.xlabel("Time [s]")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # exec_part_1()
    exec_part_2()
