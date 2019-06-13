import numpy
import numpy.linalg

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

def t_seq(t):
    t = numpy.asarray(t)
    return numpy.array([t**i for i in range(11)], dtype=numpy.float64)

def dt_seq(t):
    t = numpy.asarray(t)
    z = numpy.expand_dims(numpy.zeros(t.shape), axis=0)
    dt_ = numpy.array([i*t**(i-1) for i in range(1, 11)], dtype=numpy.float64)
    return numpy.concatenate((z, dt_))

def ddt_seq(t):
    t = numpy.asarray(t)
    z = numpy.expand_dims(numpy.zeros(t.shape), axis=0)
    dt__ = numpy.array([i*(i-1)*t**(i-2) for i in range(2, 11)], dtype=numpy.float64)
    return numpy.concatenate((z, z, dt__))

def generate_q1funcs(a_6_10):
    ti = 0.0
    tf = 3.0
    q1i = 0.0
    q1f = -numpy.pi / 6.0
    cond_i_f = numpy.array([q1i, 0.0, 0.0, q1f, 0.0, 0.0])
    Ti = t_seq(ti)
    dTi = dt_seq(ti)
    ddTi = ddt_seq(ti)
    Tf = t_seq(tf)
    dTf = dt_seq(tf)
    ddTf = ddt_seq(tf)
    T = numpy.vstack((Ti, dTi, ddTi, Tf, dTf, ddTf))
    a_0_5 = numpy.linalg.solve(T[:, :6], cond_i_f - T[:, 6:] @ a_6_10)
    a_0_10 = numpy.hstack((a_0_5, a_6_10))
    q1func = lambda t: a_0_10 @ t_seq(t)
    dq1func = lambda t: a_0_10 @ dt_seq(t)
    ddq1func = lambda t: a_0_10 @ ddt_seq(t)
    return q1func, dq1func, ddq1func

def generate_q2funcs(a_6_10):
    ti = 0.0
    tf = 3.0
    q2i = 0.0
    q2f = 2.0 * numpy.pi / 3.0
    cond_i_f = numpy.array([q2i, 0.0, 0.0, q2f, 0.0, 0.0])
    Ti = t_seq(ti)
    dTi = dt_seq(ti)
    ddTi = ddt_seq(ti)
    Tf = t_seq(tf)
    dTf = dt_seq(tf)
    ddTf = ddt_seq(tf)
    T = numpy.vstack((Ti, dTi, ddTi, Tf, dTf, ddTf))
    a_0_5 = numpy.linalg.solve(T[:, :6], cond_i_f - T[:, 6:] @ a_6_10)
    a_0_10 = numpy.hstack((a_0_5, a_6_10))
    q2func = lambda t: a_0_10 @ t_seq(t)
    dq2func = lambda t: a_0_10 @ dt_seq(t)
    ddq2func = lambda t: a_0_10 @ ddt_seq(t)
    return q2func, dq2func, ddq2func

def myf(a1_6_10, a2_6_10):
    f1, f2, f3 = generate_q1funcs(a1_6_10)
    p1, p2, p3 = generate_q2funcs(a2_6_10)
    t = numpy.linspace(0, 3, 3001)
    q1s = f1(t)
    dq1s = f2(t)
    ddq1s = f3(t)
    q2s = p1(t)
    dq2s = p2(t)
    ddq2s = p3(t)
    tau1 = m11(q1s, q2s, dq1s, dq2s, 0) * ddq1s + m12(q1s, q2s, dq1s, dq2s, 0) * ddq2s + h1(q1s, q2s, dq1s, dq2s, 0) + g1(q1s, q2s, dq1s, dq2s, 0)
    tau2 = m21(q1s, q2s, dq1s, dq2s, 0) * ddq1s + m22(q1s, q2s, dq1s, dq2s, 0) * ddq2s + h2(q1s, q2s, dq1s, dq2s, 0) + g2(q1s, q2s, dq1s, dq2s, 0)
    return tau1, tau2

if __name__ == "__main__":
    a1_6_10 = numpy.zeros(5, dtype=numpy.float64)
    a2_6_10 = numpy.zeros(5, dtype=numpy.float64)
    f1, f2, f3 = generate_q1funcs(a1_6_10)
    p1, p2, p3 = generate_q2funcs(a2_6_10)
