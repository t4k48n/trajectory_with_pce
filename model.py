import numpy
import numpy.linalg

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot

g = 9.80665

m1 = 1.0
m2 = 1.0

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

def _t_seq_slower(t):
    t = numpy.asarray(t, dtype=numpy.float64)
    return numpy.array([_t_seq_slower.tf(i, t) for i in range(11)], dtype=numpy.float64)
_t_seq_slower.tf = numpy.vectorize(lambda i, t: t ** i)

SEQ_INDEX = numpy.arange(11)

def t_seq(t):
    t = numpy.asarray(t, dtype=numpy.float64)
    if t.ndim == 0:
        return t ** SEQ_INDEX
    elif t.ndim == 1:
        return t ** numpy.expand_dims(SEQ_INDEX, 1)
    raise ValueError("dimention of t must be 0 or one")
SEQ_INDEX = numpy.arange(11)

def _dt_seq_slower(t):
    t = numpy.asarray(t, dtype=numpy.float64)
    return numpy.array([_dt_seq_slower.dtf(i, t) for i in range(11)], dtype=numpy.float64)
_dt_seq_slower.dtf = numpy.vectorize(lambda i, t: i * t ** (i - 1.0) if i - 1 >= 0 else 0.0)

def dt_seq(t):
    t = numpy.asarray(t, dtype=numpy.float64)
    if t.ndim == 0:
        return SEQ_INDEX * numpy.nan_to_num(t ** (SEQ_INDEX - 1.0))
    elif t.ndim == 1:
        seq_index = numpy.expand_dims(SEQ_INDEX, 1)
        return seq_index * numpy.nan_to_num(t ** (seq_index - 1.0))
    raise ValueError("dimention of t must be 0 or one")

def _ddt_seq_slower(t):
    t = numpy.asarray(t, dtype=numpy.float64)
    return numpy.array([_ddt_seq_slower.ddtf(i, t) for i in range(11)], dtype=numpy.float64)
_ddt_seq_slower.ddtf = numpy.vectorize(lambda i, t: i*(i-1.0)*t**(i-2.0) if i - 2 >= 0 else 0.0)

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
T_FINAL = 3.0
T_STEP_N = 3001 # 時系列[T_INIT, ..., T_FINAL]の長さ。
                # 始点、終点を含むことに注意
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
    tau1 = m11(q1s, q2s, dq1s, dq2s, 0) * ddq1s + m12(q1s, q2s, dq1s, dq2s, 0) * ddq2s + h1(q1s, q2s, dq1s, dq2s, 0) + g1(q1s, q2s, dq1s, dq2s, 0)
    tau2 = m21(q1s, q2s, dq1s, dq2s, 0) * ddq1s + m22(q1s, q2s, dq1s, dq2s, 0) * ddq2s + h2(q1s, q2s, dq1s, dq2s, 0) + g2(q1s, q2s, dq1s, dq2s, 0)
    return tau1, tau2

if __name__ == "__main__":
    a1_6_10 = numpy.array([1.0, 0.0, 0.0, 0.0, 0.0])
    a2_6_10 = numpy.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    f1, f2, f3 = generate_q1funcs(a1_6_10)
    p1, p2, p3 = generate_q2funcs(a2_6_10)

    q1 = f1(T_SERIES)
    q2 = p1(T_SERIES)

    fig = matplotlib.pyplot.figure(figsize=(10.0, 8.0))
    ax1 = fig.add_subplot(211)
    ax1.plot(q1, label="q1")
    ax1.plot(q2, label="q2")
    ax1.legend()

    tau1, tau2 = calculate_taus(a1_6_10, a2_6_10)

    ax2 = fig.add_subplot(212)
    ax2.plot(tau1, label="tau1")
    ax2.plot(tau2, label="tau2")
    ax2.legend()
    matplotlib.pyplot.show()
