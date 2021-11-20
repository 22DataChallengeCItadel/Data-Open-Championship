import numpy as np

# Import data
P_US = 1
P1_US = 3
Q_US = 10000000
Q1_US = 1000000
R = 3
C_US = 0.1
C1_US = 1
e = -0.5

# Baseline model
## Six unknowns: P_EXP, P1_EXP, Q, Q1, P_DOM, P1_DOM
## Eight parameters: P_US, P1_US, Q_US, Q1_US, R, C_US, C1_US, e

# Equations and solution
a = np.array(
    [
        [1, 0, 0, 0, -P_US / Q_US, 0],
        [0, 1, 0, 0, 0, -P1_US / Q1_US],
        [1, 0, -1, 0, 0, 0],
        [0, 1, 0, -1, 0, 0],
        [0, 0, e * Q_US / P_US, -0.5 * e * Q_US / P1_US, -1, 0],
        [0, 0, -0.5 * e * Q1_US / P_US, e * Q1_US / P1_US, 0, -1],
    ]
)
b = np.array(
    [0, 0, -R * C_US, -R * C1_US, (0.5 * e - 1) * Q_US, (0.5 * e - 1) * Q1_US]
)
x = np.linalg.solve(a, b)