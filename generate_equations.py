import sys
from sympy import *
from sympy.stats import Normal, E, variance

var("l1 l2 g t m1 m2 I1 I2 tau1 tau2 m")

q1 = Function("q1")(t)
q2 = Function("q2")(t)

x = l1 * cos(q1) + l2 * cos(q1 + q2)
y = l1 * sin(q1) + l2 * sin(q1 + q2)

x1g = l1 * cos(q1) / 2
y1g = l1 * sin(q1) / 2
x2g = l1 * cos(q1) + l2 * cos(q1 + q2) / 2
y2g = l1 * sin(q1) + l2 * sin(q1 + q2) / 2
xmg = l1 * cos(q1) + l2 * cos(q1 + q2)
ymg = l1 * sin(q1) + l2 * sin(q1 + q2)

dq1 = q1.diff(t)
dq2 = q2.diff(t)
dx1g = x1g.diff(t)
dy1g = y1g.diff(t)
dx2g = x2g.diff(t)
dy2g = y2g.diff(t)
dxmg = xmg.diff(t)
dymg = ymg.diff(t)

K1 = trigsimp((m1 * dx1g ** 2) / 2 + (m1 * dy1g ** 2) / 2 + (I1 * dq1 ** 2) / 2)
K2 = trigsimp((m2 * dx2g ** 2) / 2 + (m2 * dy2g ** 2) / 2 + (I2 * dq2 ** 2) / 2)
Km = trigsimp((m * dxmg ** 2) / 2 + (m * dymg ** 2) / 2)

U1 = m1 * g * y1g
U2 = m2 * g * y2g
Um = m * g * ymg

L = (K1 + K2 + Km) - (U1 + U2 + Um)

ddq1 = dq1.diff(t)
ddq2 = dq2.diff(t)

_eom1 = L.diff(dq1).diff(t) - L.diff(q1) - tau1
_eom2 = L.diff(dq2).diff(t) - L.diff(q2) - tau2
eom1 = _eom1.expand().collect(ddq1).collect(ddq2).collect(g)
eom2 = _eom2.expand().collect(ddq1).collect(ddq2).collect(g)

_m11, = (term for term in eom1.args if ddq1 in term.atoms(Derivative))
_m12, = (term for term in eom1.args if ddq2 in term.atoms(Derivative))
_m21, = (term for term in eom2.args if ddq1 in term.atoms(Derivative))
_m22, = (term for term in eom2.args if ddq2 in term.atoms(Derivative))

m11 = _m11 / ddq1
m12 = _m12 / ddq2
m21 = _m21 / ddq1
m22 = _m22 / ddq2
g1, = (term for term in eom1.args if g in term.atoms(Symbol))
g2, = (term for term in eom2.args if g in term.atoms(Symbol))
h1 = eom1 + tau1 - _m11 - _m12 - g1
h2 = eom2 + tau2 - _m21 - _m22 - g2

def myprint(prefix, expr, file=sys.stdout):
    expr_str = (str(expr).replace("(t)", "")
                         .replace("Derivative(q1, t)", "dq1")
                         .replace("Derivative(q2, t)", "dq2"))
    print(prefix, expr_str, file=file)

with open("equations.txt", "w") as eqf:
    myprint("x =", x, file=eqf)
    myprint("y =", y, file=eqf)

    myprint("m11 =", m11, file=eqf)
    myprint("m12 =", m12, file=eqf)
    myprint("m21 =", m21, file=eqf)
    myprint("m22 =", m22, file=eqf)
    myprint("g1 =", g1, file=eqf)
    myprint("g2 =", g2, file=eqf)
    myprint("h1 =", h1, file=eqf)
    myprint("h2 =", h2, file=eqf)
print("FINISH", file=sys.stderr)
