import math
import numpy as np
import matplotlib.pyplot as plt
from Tendon import *


class Muscle(object):
    def __init__(self):
        self.n_sarc_series = 1000
        self.n_sarc_parall = 300
        self.sarc_series_len_ratio = -1  # fiber length /  n_sarc_series
        self.sarc_series_len_opt_ratio = -1  # optimal fiber length /  n_sarc_series
        self.sarc_parall_force_ratio = -1  # max iso force /  n_sarc_parall

        self.f_m_opt = self.set_max_iso_force(3549)  # N, maximum isometric force at optimal fiber length
        self.phi_opt = 0.2  # rad, optimal pennation angle
        self.l_m_opt = 0.05  # m, optimal fiber length
        self.l_m = 0  # m, current fiber length. scalar.
        self.l_m_tilde = 0
        self.f_p_lmtilde = 0
        self.f_a_lmtilde = 0
        self.a = np.array([])  # [0, 1]. muscle activation a(u). array.

        # self.plot_force_curves()

    def __str__(self):
        return (f"Muscle Object:\n"
                f" - Maximum Isometric Force: {self.f_m_opt} N\n"
                f" - Optimal Fiber Length: {self.l_m_opt} m\n"
                f" - Current Fiber Length: {self.l_m} m\n"
                f" - Fiber ratio: {self.l_m_tilde} m\n"
                f" - F_a: {self.f_a_lmtilde} m\n"
                f" - F_p: {self.f_p_lmtilde} m\n"
                f" - Pennation Angle: {self.phi_opt} rad\n"
                )

    def plot_force_curves(self, hold_on_flag=False):
        ratio_arr = np.linspace(0.5, 1.5, 100)
        fp_arr = np.array([])
        fa_arr = np.array([])

        for ratio_curr in ratio_arr:
            fp_curr = self.compute_normed_passive_force_len(ratio_curr)
            fa_curr = Muscle.compute_normed_active_force_len(ratio_curr)
            fp_arr = np.append(fp_arr, fp_curr)
            fa_arr = np.append(fa_arr, fa_curr)

        color = 'b'
        if hold_on_flag:
            color = 'g'

        plt.plot(ratio_arr, fa_arr, color=color, linestyle='--', label="normed active force")
        plt.plot(ratio_arr, fp_arr, color=color, linestyle='-.', label="normed passive force")
        plt.plot(ratio_arr, fa_arr + fp_arr, color=color, linestyle='-', label="normed total force")
        plt.axhline(y=1, color='black', linestyle='-', label="Max isometric force")
        plt.grid(True, color="grey", linewidth="0.2", linestyle="-")
        plt.title("Normalized total (passive + active) force-length curve")
        plt.xlabel("Normalized length (m), i.e. fiber length / optimal fiber length")
        plt.ylabel("Force (N), i.e. F / max isometric force")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
        plt.xticks(np.arange(0.5, 1.6, 0.1))
        plt.ylim(0, 1.5)  # Set y-axis limits to 0 and 60

        if not hold_on_flag:
            plt.show()

    def plot_force_curves_2(self, hold_on_flag=False):
        ratio_arr = np.linspace(0.01, 0.2, 100)
        fp_arr = np.array([])
        fa_arr = np.array([])

        for ratio_curr in ratio_arr:
            fp_curr = self.compute_normed_passive_force_len(ratio_curr / self.l_m_opt)
            fa_curr = Muscle.compute_normed_active_force_len(ratio_curr / self.l_m_opt)
            fp_arr = np.append(fp_arr, fp_curr)
            fa_arr = np.append(fa_arr, fa_curr)

        color = 'b'
        if hold_on_flag:
            color = 'g'

        plt.plot(ratio_arr, fa_arr, color=color, linestyle='--', label="normed active force")
        plt.plot(ratio_arr, fp_arr, color=color, linestyle='-.', label="normed passive force")
        plt.plot(ratio_arr, fa_arr + fp_arr, color=color, linestyle='-', label="normed total force")
        plt.axhline(y=1, color='black', linestyle='-', label="Max isometric force")
        plt.grid(True, color="grey", linewidth="0.2", linestyle="-")
        plt.title("Normalized total (passive + active) force-length curve")
        plt.xlabel("Normalized length (m), i.e. fiber length / optimal fiber length")
        plt.ylabel("Force (N), i.e. F / max isometric force")
        plt.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
        plt.xticks(np.arange(0.01, 0.2, 0.02))
        plt.ylim(0, 1.5)  # Set y-axis limits to 0 and 60

        if not hold_on_flag:
            plt.show()

    def compute_f_m(self, normed_flag=False, plot_time_vector=None):
        f_m_a = self.compute_f_m_a()
        f_m_p = self.compute_f_m_p()
        f_m = f_m_a + f_m_p

        if normed_flag:
            f_m = f_m / self.f_m_opt

        if plot_time_vector is not None:
            plt.plot(plot_time_vector, f_m)
            plt.title("Muscle Force")
            plt.grid(True, color="grey", linewidth="0.4", linestyle="-")
            plt.show()

        print(f"max f_m value: {np.max(f_m)}")
        return f_m

    def compute_f_m_p(self):
        self.l_m_tilde = self.l_m / self.l_m_opt  # ratio between current fiber len and opt. fiber len
        self.f_p_lmtilde = self.compute_normed_passive_force_len(self.l_m_tilde)  # active force-len function
        f_m_p = self.f_p_lmtilde * self.f_m_opt
        return f_m_p

    def compute_f_m_a(self):
        self.l_m_tilde = self.l_m / self.l_m_opt  # ratio between current fiber len and opt. fiber len
        self.f_a_lmtilde = Muscle.compute_normed_active_force_len(self.l_m_tilde)  # active force-len function
        f_m_a = self.f_a_lmtilde * self.f_m_opt * self.a
        return f_m_a

    def compute_normed_passive_force_len(self, fiber_len_ratio):
        """
        formula from Buchanan 2004 "Neuromusculoskeletal Modeling: Estimation of Muscle Forces  and Joint Moments and Movements From Measurements of Neural Command"
        check plot with Zajac 1989 "Muscle and tendon: properties, models, scaling, and application to biomechanics and motor control"

        :param fiber_len_ratio: l_m / l_m_o
        :return: normalized passive force
        """
        if fiber_len_ratio < 1:
            # print(f"no passive force at {fiber_len_ratio}")
            return 0

        fp = np.exp(10 * fiber_len_ratio - 1)
        fp = fp / np.exp(5.5)
        fp = fp / self.f_m_opt  # normalize wrt max isometric force

        # print(f"compute_normed_passive_force_len. ratio: {fiber_len_ratio:.2f}, fp: {fp}")
        return fp

    @staticmethod
    def compute_normed_active_force_len(fiber_len_ratio):
        if fiber_len_ratio > 1.5 or fiber_len_ratio < 0.5:
            return 0

        fa = -4 * (fiber_len_ratio - 1) * (fiber_len_ratio - 1) + 1  # wiki?
        # fa = fa / self.f_m_opt  # no need to normalize. [0, 1] range is implicit in modeling function.
        # print(f"compute_normed_active_force_len. ratio: {fiber_len_ratio:.2f}, fa: {fa}")
        return fa

    # getters and setters
    def set_fiber_len(self, l_mt):
        """
        :param l_mt, muscle-tendon length [m]
        :return: computed fiber length [m]
        """
        term_1 = self.l_m_opt * math.sin(self.phi_opt)
        term_1 = term_1 * term_1

        term_2 = l_mt - Tendon.get_len()
        term_2 = term_2 * term_2

        self.l_m = math.sqrt(term_1 + term_2)
        self.sarc_series_len_ratio = self.l_m / self.n_sarc_series
        self.sarc_series_len_opt_ratio = self.l_m_opt / self.n_sarc_series

        return self.l_m

    def get_fiber_len(self):
        """
        :return: computed fiber length [m]
        """
        return self.l_m

    def set_max_iso_force(self, new_f_m_opt):
        self.f_m_opt = new_f_m_opt
        self.sarc_parall_force_ratio = self.f_m_opt / self.n_sarc_parall
        return self.f_m_opt

    def get_max_iso_force(self):
        return self.f_m_opt

    def update_n_sarc_series(self, gain):
        self.n_sarc_series = self.n_sarc_series * gain
        self.l_m_opt = self.n_sarc_series * self.sarc_series_len_opt_ratio
        self.l_m = self.n_sarc_series * self.sarc_series_len_ratio

    def update_n_sarc_parall(self, gain):
        self.n_sarc_parall = self.n_sarc_parall * gain
        self.f_m_opt = self.n_sarc_parall * self.sarc_parall_force_ratio

    def get_opt_fiber_len(self):
        return self.l_m_opt

    def set_activation(self, curr_activation):
        """
        :param curr_activation: numpy array containing muscle activation values
        """
        if curr_activation.any() > 1 or curr_activation.any() < 0:
            print(f"[set_activation] activation must be between 0 and 1. current: {curr_activation}\n")
            exit(1)

        self.a = curr_activation
