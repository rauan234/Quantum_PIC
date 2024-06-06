import numpy as np
import random

# ------------------------------------------------------------------------------
#                                 PARTICLE
# ------------------------------------------------------------------------------


class Particle:
    def __init__(self, m, q, R, v=None, u=None):
        assert isinstance(m, float) and m > 0
        assert isinstance(q, float)
        assert isinstance(R, np.ndarray) and R.shape == (3,)
        self.m = m  # mass
        self.q = q  # charge

        self.R = R.copy()     # position
        self.v = np.zeros(3)  # velocity
        self.u = np.zeros(3)  # reduced momentum, u = p / m = gamma v

        if not isinstance(v, type(None)):
            assert isinstance(v, np.ndarray) and v.shape == (3,)
            assert np.linalg.norm(v) < 1.  # going slower than light
            self.v = v.copy()
            self.compute_u()
        elif not isinstance(u, type(None)):
            assert isinstance(u, np.ndarray) and u.shape == (3,)
            self.u = u.copy()
            self.compute_v()

    def copy(self):
        ptc = Particle(self.m, self.q, self.R)
        ptc.v = self.v.copy()
        ptc.u = self.u.copy()
        return ptc

    def compute_u(self):
        self.u = self.v / np.sqrt(1 - self.v[0]**2 - self.v[1]**2 - self.v[2]**2)

    def compute_v(self):
        self.v = self.u / np.sqrt(1 + self.u[0]**2 + self.u[1]**2 + self.u[2]**2)

    # the Vay particle pusher
    def push(self, dt, E, B):
        g = self.q * dt / self.m

        u_pr = self.u + g * (E + 0.5 * np.cross(self.v, B))  # u'            vector
        gamma_pr_sq = 1 + np.dot(u_pr, u_pr)                 # (gamma')^2    scalar
        tau = 0.5 * g * B                                    # tau           vector
        sigma = gamma_pr_sq - np.dot(tau, tau)               # sigma         scalar
        u_star = np.dot(u_pr, tau)                           # u^*           scalar
        gamma_new = np.sqrt(0.5 * (sigma +                   # gamma_new     scalar
            np.sqrt(sigma**2 + 4*(np.dot(tau, tau) + u_star**2))
        ))
        t = tau / gamma_new                                  # t             vector
        s = 1 / (1 + np.dot(t, t))                           # s             scalar

        self.u = s * (u_pr + np.dot(u_pr, t) * t + np.cross(u_pr, t))
        self.v = self.u / gamma_new
        self.R += dt * self.v


# create a randomized particle
def random_particle(R_range, u_range=10., q_range=1., m_range=(0.1, 1.)):
    # generate a particle whose location is chosen at random from [0, R_range[0]] cross ... cross [0, R_range[2]]
    # whose reduced momentum p/m is chosen at random from [-u_range, u_range]^3,
    # and whose charge and mass are chosen at random from [-q_range, q_range] and [m_range[0], m_range[1]]
    assert isinstance(R_range, tuple) and len(R_range) == 3 and all(map(lambda x: x > 0, R_range))
    assert isinstance(u_range, float) and u_range > 0
    assert isinstance(q_range, float) and q_range > 0
    (m_min, m_max) = m_range
    assert isinstance(m_min, float) and isinstance(m_max, float) and 0 < m_min < m_max

    R = np.array(R_range) * np.random.rand(3)
    u = u_range * (-1. + 2. * np.random.rand(3))
    q = q_range * (-1. + 2. * random.random())
    m = m_min + (m_max - m_min) * random.random()

    return Particle(m, q, R, u=u)
