"""
Equations of motion for a cart with a series-linked TRIPLE
pendulum (point mass at the end of each link), derived via Lagrangian 
mechanics.

State: x (cart pos), th1, th2, th3 (link angles from upward vertical),
       xd, th1d, th2d, th3d (velocities)
Input: F (force on cart)
Params: M (cart mass), m1, m2, m3 (point masses), l1, l2, l3 (link lengths),
        g (gravity), b (cart damping/friction coefficient)

"""
import numpy as np

def accelerations(x, th1, th2, th3, xd, th1d, th2d, th3d, F, M, m1, m2, m3, l1, l2, l3, g, b):
    # Precompute trig
    s1, c1 = np.sin(th1), np.cos(th1)
    s2, c2 = np.sin(th2), np.cos(th2)
    s3, c3 = np.sin(th3), np.cos(th3)
    s12, c12 = np.sin(th1 - th2), np.cos(th1 - th2)
    s13, c13 = np.sin(th1 - th3), np.cos(th1 - th3)
    s23, c23 = np.sin(th2 - th3), np.cos(th2 - th3)

    # 4x4 Mass Matrix
    M_mat = np.array([
        [M + m1 + m2 + m3,                 (m1 + m2 + m3)*l1*c1,  (m2 + m3)*l2*c2,        m3*l3*c3        ],
        [(m1 + m2 + m3)*l1*c1,             (m1 + m2 + m3)*l1**2,  (m2 + m3)*l1*l2*c12,    m3*l1*l3*c13    ],
        [(m2 + m3)*l2*c2,                  (m2 + m3)*l1*l2*c12,   (m2 + m3)*l2**2,        m3*l2*l3*c23    ],
        [m3*l3*c3,                         m3*l1*l3*c13,          m3*l2*l3*c23,           m3*l3**2         ]
    ])

    # 4x1 Forcing Vector (Coriolis, Centripetal, Gravity, External Force)
    C_vec = np.array([
        F - b*xd + (m1 + m2 + m3)*l1*th1d**2*s1 + (m2 + m3)*l2*th2d**2*s2 + m3*l3*th3d**2*s3,
        (m1 + m2 + m3)*g*l1*s1 - (m2 + m3)*l1*l2*th2d**2*s12 - m3*l1*l3*th3d**2*s13,
        (m2 + m3)*g*l2*s2 + (m2 + m3)*l1*l2*th1d**2*s12 - m3*l2*l3*th3d**2*s23,
        m3*g*l3*s3 + m3*l1*l3*th1d**2*s13 + m3*l2*l3*th2d**2*s23
    ])

    # Solve M * accelerations = C
    try:
        acc = np.linalg.solve(M_mat, C_vec)
    except np.linalg.LinAlgError:
        acc = np.zeros(4)

    # Return xdd, th1dd, th2dd, th3dd
    return acc[0], acc[1], acc[2], acc[3]
