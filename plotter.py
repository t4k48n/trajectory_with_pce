# Proceeding向けの適当なグラフをプロット
import pathlib

import numpy

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.rc("font", size=10)

def sort_key_for_npy(path):
    stem = path.stem
    parts = stem.split('_')
    if parts[2] == 'init':
        return ('init', parts[1])
    base, exp = parts[3].split('E')
    return ('opt', exp, base, parts[1])

def get_weight_string(path):
    stem = path.stem
    parts = stem.split('_')
    return parts[-1]

def get_tau(param_q1, param_q2):
    import model
    a1_6_10, a2_6_10 = param_q1[6:], param_q2[6:]
    tau1, tau2 = model.calculate_taus(a1_6_10, a2_6_10)
    return tau1, tau2

def get_var(param_q1, param_q2):
    import model
    a1_6_10, a2_6_10 = param_q1[6:], param_q2[6:]
    tau1, tau2 = model.calculate_taus(a1_6_10, a2_6_10)
    q1_pc, q2_pc, _, _ = model.calculate_pc(tau1, tau2)
    q1_pc_var_last = model.calculate_pc_var(q1_pc[:, -1])
    q2_pc_var_last = model.calculate_pc_var(q2_pc[:, -1])
    return q1_pc_var_last, q2_pc_var_last

def plot_all_weights_vars():
    cd = pathlib.Path(__file__).absolute().parent
    q1_npy_files = sorted(cd.glob('param_q1_*.npy'), key=sort_key_for_npy)
    q2_npy_files = sorted(cd.glob('param_q2_*.npy'), key=sort_key_for_npy)
    #[print(npy) for npy in npy_files]
    q1_npy_data = [numpy.load(npy) for npy in q1_npy_files]
    q2_npy_data = [numpy.load(npy) for npy in q2_npy_files]

    npy_pc_taus_path = cd / "npy_pc_taus.npy"
    if not npy_pc_taus_path.exists():
        npy_pc_taus = numpy.asarray([get_tau(pq1, pq2)
                                     for pq1, pq2
                                     in zip(q1_npy_data, q2_npy_data)])
        numpy.save(npy_pc_taus_path, npy_pc_taus)
    else:
        npy_pc_taus = numpy.load(npy_pc_taus_path)

    npy_pc_vars_path = cd / "npy_pc_vars.npy"
    if not npy_pc_vars_path.exists():
        npy_pc_vars = numpy.asarray([get_var(pq1, pq2)
                                     for pq1, pq2
                                     in zip(q1_npy_data, q2_npy_data)])
        numpy.save(npy_pc_vars_path, npy_pc_vars)
    else:
        npy_pc_vars = numpy.load(npy_pc_vars_path)
    x_labels = [get_weight_string(npy) for npy in q1_npy_files]
    plt.plot(x_labels, npy_pc_vars[:, 0])
    plt.plot(x_labels, npy_pc_vars[:, 1])
    plt.show()

def make_pair_list(single_list):
    if len(single_list) % 2 != 0:
        raise ValueError("length of list must be even")
    return [(single_list[2 * i], single_list[2 * i + 1])
            for i in range(len(single_list) // 2)]

def plot_selected():
    import model
    cd = pathlib.Path(__file__).absolute().parent
    path_weight_list = [(cd / "param_q1_opt_1E-3.npy", "1E-3"),
                        (cd / "param_q2_opt_1E-3.npy", "1E-3"),
                        (cd / "param_q1_opt_1E-4.npy", "1E-4"),
                        (cd / "param_q2_opt_1E-4.npy", "1E-4"),
                        (cd / "param_q1_opt_1E-5.npy", "1E-5"),
                        (cd / "param_q2_opt_1E-5.npy", "1E-5"),
                        (cd / "param_q1_opt_1E-6.npy", "1E-6"),
                        (cd / "param_q2_opt_1E-6.npy", "1E-6")]
    param_pair_weight_list = [(numpy.load(p1), numpy.load(p2), w)
                              for (p1, w), (p2, _)
                              in make_pair_list(path_weight_list)]
    x_ticks = [w for _, _, w in param_pair_weight_list]
    pc_vars = numpy.array([get_var(p1, p2) for p1, p2, _ in param_pair_weight_list])
    pc_q1_std = numpy.sqrt(pc_vars[:, 0])
    pc_q2_std = numpy.sqrt(pc_vars[:, 1])

    height = 55 / 25.4
    width = 84 / 25.4

    plt.figure(figsize=(width, height))
    plt.subplots_adjust(top=0.939, bottom=0.207, left=0.202, right=0.982, hspace=0.2, wspace=0.2)
    plt.plot(x_ticks, pc_q1_std, linestyle="none", marker="o", markerfacecolor="#00000000", markeredgecolor="#000000ff", label=r"$\theta_1$")
    plt.plot(x_ticks, pc_q2_std, linestyle="none", marker="x", markerfacecolor="#00000000", markeredgecolor="#000000ff", label=r"$\theta_2$")
    plt.xlabel(r"$w_{\tau}$")
    plt.ylabel(r"Standard deviation [deg]")
    plt.grid()
    plt.legend()

    plt.figure(figsize=(width, height))
    plt.subplots_adjust(top=0.729, bottom=0.207, left=0.202, right=0.982, hspace=0.2, wspace=0.2)
    tlist = numpy.linspace(0, 3, 1001)
    p1, p2, w = param_pair_weight_list[0]
    tau1, tau2 = get_tau(p1, p2)
    plt.plot(tlist, tau1, linestyle="-", color="black", label=r"$\tau_{{1}}$ ({})".format(w))
    plt.plot(tlist, tau2, linestyle="-.", color="black", label=r"$\tau_{{2}}$ ({})".format(w))
    p1, p2, w = param_pair_weight_list[2]
    tau1, tau2 = get_tau(p1, p2)
    plt.plot(tlist, tau1, linestyle="--", color="black", label=r"$\tau_{{1}}$ ({})".format(w))
    plt.plot(tlist, tau2, linestyle=":", color="black", label=r"$\tau_{{2}}$ ({})".format(w))
    plt.xlabel(r"Time [s]")
    plt.ylabel(r"Torque [Nm]")
    plt.grid()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    plt.figure(figsize=(width, height))
    plt.subplots_adjust(top=0.729, bottom=0.207, left=0.202, right=0.982, hspace=0.2, wspace=0.2)
    tlist = numpy.linspace(0, 3, 1001)
    p1, p2, w = param_pair_weight_list[0]
    tau1, tau2 = get_tau(p1, p2)
    q1, q2, _, _ = model.simulate(tau1, tau2, 0)
    plt.plot(tlist, q1 * 180 / numpy.pi, linestyle="-", color="black", label=r"$\theta_{{1}}$ ({})".format(w))
    plt.plot(tlist, q2 * 180 / numpy.pi, linestyle="-.", color="black", label=r"$\theta_{{2}}$ ({})".format(w))
    p1, p2, w = param_pair_weight_list[2]
    tau1, tau2 = get_tau(p1, p2)
    q1, q2, _, _ = model.simulate(tau1, tau2, 0)
    plt.plot(tlist, q1 * 180 / numpy.pi, linestyle="--", color="black", label=r"$\theta_{{1}}$ ({})".format(w))
    plt.plot(tlist, q2 * 180 / numpy.pi, linestyle=":", color="black", label=r"$\theta_{{2}}$ ({})".format(w))
    plt.xlabel(r"Time [s]")
    plt.ylabel(r"Posture angle [deg]")
    plt.yticks([-60, -30, 0, 30, 60, 90, 120])
    plt.grid()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    plt.show()

if __name__ == '__main__':
    plot_selected()
