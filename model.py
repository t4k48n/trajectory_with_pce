import numpy
import numpy.random
import numpy.linalg

import scipy.special
import scipy.optimize

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot

g = 9.80665

# リンク質量
m1 = 1.0
m2 = 1.0

# 把持物質量（期待値と不確かさの大きさ）
mm = 0.1
dm = 0.001
m = lambda z: mm + dm * z

l1 = 1.0
l2 = 1.0

I1 = m1 ** l1 / 12.0
I2 = m2 ** l2 / 12.0

def m11(q1, q2, dq1, dq2, m):
    return I1 + l1**2*m1/4 + l1**2*m2 + l1**2*numpy.sin(q1)**2*m + l1**2*numpy.cos(q1)**2*m + l1*l2*m2*numpy.cos(q2) + 2*l1*l2*numpy.sin(q1 + q2)*numpy.sin(q1)*m + 2*l1*l2*numpy.cos(q1 + q2)*numpy.cos(q1)*m + l2**2*m2/4 + l2**2*numpy.sin(q1 + q2)**2*m + l2**2*numpy.cos(q1 + q2)**2*m

def m12(q1, q2, dq1, dq2, m):
    return l1*l2*m2*numpy.cos(q2)/2 + l1*l2*numpy.sin(q1 + q2)*numpy.sin(q1)*m + l1*l2*numpy.cos(q1 + q2)*numpy.cos(q1)*m + l2**2*m2/4 + l2**2*numpy.sin(q1 + q2)**2*m + l2**2*numpy.cos(q1 + q2)**2*m

def m21(q1, q2, dq1, dq2, m):
    return l1*l2*m2*numpy.cos(q2)/2 + l1*l2*numpy.sin(q1 + q2)*numpy.sin(q1)*m + l1*l2*numpy.cos(q1 + q2)*numpy.cos(q1)*m + l2**2*m2/4 + l2**2*numpy.sin(q1 + q2)**2*m + l2**2*numpy.cos(q1 + q2)**2*m

def m22(q1, q2, dq1, dq2, m):
    return I2 + l2**2*m2/4 + l2**2*numpy.sin(q1 + q2)**2*m + l2**2*numpy.cos(q1 + q2)**2*m

def g1(q1, q2, dq1, dq2, m):
    return g*(l1*m1*numpy.cos(q1)/2 + l1*m2*numpy.cos(q1) + l1*numpy.cos(q1)*m + l2*m2*numpy.cos(q1 + q2)/2 + l2*numpy.cos(q1 + q2)*m)

def g2(q1, q2, dq1, dq2, m):
    return g*(l2*m2*numpy.cos(q1 + q2)/2 + l2*numpy.cos(q1 + q2)*m)

def h1(q1, q2, dq1, dq2, m):
    return (-l1*l2*m2*numpy.sin(q2) - 2*l1*l2*numpy.sin(q1 + q2)*numpy.cos(q1)*m + 2*l1*l2*numpy.sin(q1)*numpy.cos(q1 + q2)*m)*dq1*dq2 + (-l1*l2*m2*numpy.sin(q2)/2 - l1*l2*numpy.sin(q1 + q2)*numpy.cos(q1)*m + l1*l2*numpy.sin(q1)*numpy.cos(q1 + q2)*m)*dq2**2

def h2(q1, q2, dq1, dq2, m):
    return (l1*l2*m2*numpy.sin(q2)/2 + l1*l2*numpy.sin(q1 + q2)*numpy.cos(q1)*m - l1*l2*numpy.sin(q1)*numpy.cos(q1 + q2)*m)*dq1**2

SEQ_INDEX = numpy.arange(11)

def t_seq(t):
    t = numpy.asarray(t, dtype=numpy.float64)
    if t.ndim == 0:
        return t ** SEQ_INDEX
    elif t.ndim == 1:
        return t ** numpy.expand_dims(SEQ_INDEX, 1)
    raise ValueError("dimention of t must be 0 or one")
SEQ_INDEX = numpy.arange(11)

def dt_seq(t):
    t = numpy.asarray(t, dtype=numpy.float64)
    if t.ndim == 0:
        return SEQ_INDEX * numpy.nan_to_num(t ** (SEQ_INDEX - 1.0))
    elif t.ndim == 1:
        seq_index = numpy.expand_dims(SEQ_INDEX, 1)
        return seq_index * numpy.nan_to_num(t ** (seq_index - 1.0))
    raise ValueError("dimention of t must be 0 or one")

def ddt_seq(t):
    t = numpy.asarray(t, dtype=numpy.float64)
    if t.ndim == 0:
        return SEQ_INDEX * (SEQ_INDEX - 1.0) * numpy.nan_to_num(t ** (SEQ_INDEX - 2.0))
    elif t.ndim == 1:
        seq_index = numpy.expand_dims(SEQ_INDEX, 1)
        return seq_index * (seq_index - 1.0) * numpy.nan_to_num(t ** (seq_index - 2.0))
    raise ValueError("dimention of t must be 0 or one")

# 時刻
T_INIT = 0.0
T_FINAL = 1.0
T_STEP_N = 1001 # 時系列[T_INIT, ..., T_FINAL]の長さ。
                # 始点、終点を含むことに注意
DELTA_T = (T_FINAL - T_INIT) / (T_STEP_N - 1)
T_SERIES = numpy.linspace(T_INIT, T_FINAL, T_STEP_N)

# 初期姿勢、終端姿勢
Q1_INIT = 0.0
Q1_FINAL = -numpy.pi / 6.0
Q2_INIT = 0.0
Q2_FINAL = 2.0 * numpy.pi / 3.0

# 境界値条件
BOUND_COND_VEC = numpy.array([t_seq(T_INIT),
                              dt_seq(T_INIT),
                              ddt_seq(T_INIT),
                              t_seq(T_FINAL),
                              dt_seq(T_FINAL),
                              ddt_seq(T_FINAL)])
Q1_BOUND_COND = numpy.array([Q1_INIT, 0.0, 0.0, Q1_FINAL, 0.0, 0.0])
Q2_BOUND_COND = numpy.array([Q2_INIT, 0.0, 0.0, Q2_FINAL, 0.0, 0.0])

def generate_q1funcs(a_6_10):
    a_0_5 = numpy.linalg.solve(BOUND_COND_VEC[:, :6], Q1_BOUND_COND - BOUND_COND_VEC[:, 6:] @ a_6_10)
    a_0_10 = numpy.hstack((a_0_5, a_6_10))
    q1func = lambda t: a_0_10 @ t_seq(t)
    dq1func = lambda t: a_0_10 @ dt_seq(t)
    ddq1func = lambda t: a_0_10 @ ddt_seq(t)
    return q1func, dq1func, ddq1func

def generate_q2funcs(a_6_10):
    a_0_5 = numpy.linalg.solve(BOUND_COND_VEC[:, :6], Q2_BOUND_COND - BOUND_COND_VEC[:, 6:] @ a_6_10)
    a_0_10 = numpy.hstack((a_0_5, a_6_10))
    q2func = lambda t: a_0_10 @ t_seq(t)
    dq2func = lambda t: a_0_10 @ dt_seq(t)
    ddq2func = lambda t: a_0_10 @ ddt_seq(t)
    return q2func, dq2func, ddq2func

def calculate_taus(a1_6_10, a2_6_10):
    q1func, dq1func, ddq1func = generate_q1funcs(a1_6_10)
    q2func, dq2func, ddq2func = generate_q2funcs(a1_6_10)
    q1s = q1func(T_SERIES)
    dq1s = dq1func(T_SERIES)
    ddq1s = ddq1func(T_SERIES)
    q2s = q2func(T_SERIES)
    dq2s = dq2func(T_SERIES)
    ddq2s = ddq2func(T_SERIES)
    tau1 = m11(q1s, q2s, dq1s, dq2s, m(0)) * ddq1s + m12(q1s, q2s, dq1s, dq2s, m(0)) * ddq2s + h1(q1s, q2s, dq1s, dq2s, m(0)) + g1(q1s, q2s, dq1s, dq2s, m(0))
    tau2 = m21(q1s, q2s, dq1s, dq2s, m(0)) * ddq1s + m22(q1s, q2s, dq1s, dq2s, m(0)) * ddq2s + h2(q1s, q2s, dq1s, dq2s, m(0)) + g2(q1s, q2s, dq1s, dq2s, m(0))
    return tau1, tau2

def calculate_ddqs(q1, q2, dq1, dq2, m, tau1, tau2):
    M = numpy.array([[m11(q1, q2, dq1, dq2, m), m12(q1, q2, dq1, dq2, m)],
                     [m21(q1, q2, dq1, dq2, m), m22(q1, q2, dq1, dq2, m)]])
    H = numpy.array([h1(q1, q2, dq1, dq2, m),
                     h2(q1, q2, dq1, dq2, m)])
    G = numpy.array([g1(q1, q2, dq1, dq2, m),
                     g2(q1, q2, dq1, dq2, m)])
    T = numpy.array([tau1, tau2])
    ddqs = numpy.linalg.solve(M, T - H - G)
    return ddqs

def simulate(tau1_series, tau2_series, z):
    #tau_series = numpy.vstack((tau1_series, tau2_series)).T
    q1_series = numpy.zeros(T_STEP_N)
    q2_series = numpy.zeros(T_STEP_N)
    dq1_series = numpy.zeros(T_STEP_N)
    dq2_series = numpy.zeros(T_STEP_N)
    q1_series[0] = Q1_INIT
    q2_series[0] = Q2_INIT
    for i in range(T_STEP_N - 1):
        dq1 = dq1_series[i]
        dq2 = dq2_series[i]
        ddq1, ddq2 = calculate_ddqs(q1_series[i], q2_series[i], dq1_series[i], dq2_series[i], m(z), tau1_series[i], tau2_series[i])
        q1_series[i + 1] += q1_series[i] + DELTA_T * dq1
        q2_series[i + 1] += q2_series[i] + DELTA_T * dq2
        dq1_series[i + 1] += dq1_series[i] + DELTA_T * ddq1
        dq2_series[i + 1] += dq2_series[i] + DELTA_T * ddq2
    return q1_series, q2_series, dq1_series, dq2_series

# PCE uses polynomials `${\phi_0, \phi_1, \ldots, \phi_{PCE_TERM_NUM - 1}}$`.
PCE_TERM_NUM = 9 # PCE_TERM_NUM = 9にしておけばほぼずれはなくなる
POLYNOMIALS = lambda z: numpy.array([scipy.special.eval_hermitenorm(i, z)
                                     for i in range(PCE_TERM_NUM)])
INNER_PRODUCTS = numpy.sqrt(2.0 * numpy.pi) * numpy.array([numpy.math.factorial(i) for i in range(PCE_TERM_NUM)])
COLLOCATIONS = numpy.asarray(sorted(scipy.special.roots_hermitenorm(PCE_TERM_NUM)[0],
                                    key=lambda c: numpy.abs(c)))
PCE_MATRIX = POLYNOMIALS(COLLOCATIONS).T

def call_pce_function(coef, z):
    coef = numpy.asarray(coef)
    z = numpy.asarray(z)
    if z.ndim == 0:
        return coef @ POLYNOMIALS(z)
    elif z.ndim == 1:
        return coef @ POLYNOMIALS(z).T
    raise ValueError("pce_function: dimention of z must be 0 or one")

def calculate_pc(tau1, tau2):
    simulate_results = [simulate(tau1, tau2, z) for z in COLLOCATIONS]
    q1_mat = numpy.array([r[0] for r in simulate_results])
    q2_mat = numpy.array([r[1] for r in simulate_results])
    dq1_mat = numpy.array([r[2] for r in simulate_results])
    dq2_mat = numpy.array([r[3] for r in simulate_results])
    q1_pc = numpy.linalg.solve(PCE_MATRIX, q1_mat)
    q2_pc = numpy.linalg.solve(PCE_MATRIX, q2_mat)
    dq1_pc = numpy.linalg.solve(PCE_MATRIX, dq1_mat)
    dq2_pc = numpy.linalg.solve(PCE_MATRIX, dq2_mat)
    return q1_pc, q2_pc, dq1_pc, dq2_pc

def calculate_pc_var(coef):
    return (coef[1:] ** 2.0) @ INNER_PRODUCTS[1:]

def evaluate_final_state(a1_6_10_and_a2_6_10):
    a1_6_10_and_a2_6_10 = numpy.asarray(a1_6_10_and_a2_6_10)
    a1_6_10 = a1_6_10_and_a2_6_10[:5]
    a2_6_10 = a1_6_10_and_a2_6_10[5:]
    tau1_series, tau2_series =  calculate_taus(a1_6_10, a2_6_10)
    q1_pc, q2_pc, dq1_pc, dq2_pc = calculate_pc(tau1_series, tau2_series)
    q1_var = calculate_pc_var(q1_pc[:, -1])
    q2_var = calculate_pc_var(q2_pc[:, -1])
    return numpy.sqrt(q1_var + q2_var)

if __name__ == "__main__":
    a1_6_10_init = numpy.zeros(5, dtype=numpy.float64)
    a2_6_10_init = numpy.zeros(5, dtype=numpy.float64)
    as_6_10_init = numpy.hstack((a1_6_10_init, a2_6_10_init))
    as_6_10_opt = scipy.optimize.fmin(evaluate_final_state, as_6_10_init)
    a1_6_10_opt, a2_6_10_opt = as_6_10_opt[:5], as_6_10_opt[5:]
