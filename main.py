import numpy as np
import random


class Grid:
    def __init__(self, mesh_size, phys_size):
        assert isinstance(phys_size, tuple) and len(phys_size) == 3
        assert all(map(lambda x: isinstance(x, float), phys_size))

        assert isinstance(mesh_size, tuple) and len(mesh_size) == 3
        assert all(map(lambda x: isinstance(x, int) and x > 0, mesh_size))

        self.phys_size = phys_size  # (xsize, ysize, zsize), say in meters
        self.mesh_size = mesh_size  # (N_1, N_2, N_3), number of mesh points in each dimension

        (self.N1, self.N2, self.N3) = self.mesh_size
        (self.Dx, self.Dy, self.Dz) = np.divide(phys_size, mesh_size)
        self.DV = self.Dx * self.Dy  * self.Dz

        # (1/dx, 1/dy, 1/dz)
        self.k0 = np.divide(mesh_size, phys_size)
        self.one_over_dV = self.k0[0] * self.k0[1] * self.k0[2]

        # initialize lapl_dft, which encodes how the
        # discrete Fourier transform of a function relates to the original DFT
        arr1 = -4 / (self.Dx ** 2) * (np.sin(np.pi / self.N1 * np.array(range(0, self.N1))) ** 2)
        arr2 = -4 / (self.Dy ** 2) * (np.sin(np.pi / self.N2 * np.array(range(0, self.N2))) ** 2)
        arr3 = -4 / (self.Dz ** 2) * (np.sin(np.pi / self.N3 * np.array(range(0, self.N3))) ** 2)
        self.lapl_dft = (arr1[:, np.newaxis, np.newaxis] +
               arr2[np.newaxis, :, np.newaxis] +
               arr3[np.newaxis, np.newaxis, :])
        self.lapl_dft[0][0][0] = 1.  # set (0, 0, 0) component to 1. because normally it is 0. (see poisson_solve)

        self.E = np.zeros((3,) + mesh_size)
        self.B = np.zeros((3,) + mesh_size)

        self.rho = np.zeros(mesh_size)
        self.J = np.zeros((3,) + mesh_size)

    def compute_form_factor(self, R):
        # compute the form factor S_r(R) for all values of r,
        # for a super-particle which is currently at position R
        assert isinstance(R, np.ndarray) and R.shape == (3,)

        def get_spline_data(x, i, N):
            # in: x a real number, i an integer such that |x - i| < 0.5, N a positive integer
            # out: the value of the quadratic spline at i-1, i, i+1
            im = (i - 1) % N
            i_ = i % N
            ip = (i + 1) % N
            d = x - i
            return ((im, 0.5 * (d - 0.5)**2),
                    (i_, 0.75 - d**2),
                    (ip, 0.5 * (d + 0.5)**2))

        S = np.zeros(self.mesh_size)

        Rk = (R[0] * self.k0[0], R[1] * self.k0[1], R[2] * self.k0[2])
        (i0, j0, k0) = tuple(map(round, Rk))

        # first, deposit charge densities
        for (i, fx) in get_spline_data(Rk[0], i0, self.mesh_size[0]):
            for (j, fy) in get_spline_data(Rk[1], j0, self.mesh_size[1]):
                for (k, fz) in get_spline_data(Rk[2], k0, self.mesh_size[2]):
                    S[i][j][k] += fx * fy * fz * self.one_over_dV

        return S

    def refresh_charge(self):
        self.rho.fill(0)
        self.J.fill(0)

    # NEED TO OPTIMIZE
    def deposit_charge(self, q, R, v, dt):
        # update the charge and current densities due to a
        # super-particle of charge q at position R and speed v,
        # over time step dt
        assert isinstance(q, float)
        assert isinstance(R, np.ndarray) and R.shape == (3,)
        assert isinstance(v, np.ndarray) and v.shape == (3,) and np.linalg.norm(v) <= 1.0
        assert isinstance(dt, float) and dt > 0

        (dx, dy, dz) = dt * v

        S000 = self.compute_form_factor(R + np.array([0, 0, 0]))
        S001 = self.compute_form_factor(R + np.array([0, 0, dz]))
        S010 = self.compute_form_factor(R + np.array([0, dy, 0]))
        S011 = self.compute_form_factor(R + np.array([0, dy, dz]))
        S100 = self.compute_form_factor(R + np.array([dx, 0, 0]))
        S101 = self.compute_form_factor(R + np.array([dx, 0, dz]))
        S110 = self.compute_form_factor(R + np.array([dx, dy, 0]))
        S111 = self.compute_form_factor(R + np.array([dx, dy, dz]))

        self.rho += q * S000

        W1 = (S100 - S000) / 3. + (S110 - S010) / 6. + (S101 - S001) / 6. + (S111 - S011) / 3.
        W2 = (S010 - S000) / 3. + (S011 - S001) / 6. + (S110 - S100) / 6. + (S111 - S101) / 3.
        W3 = (S001 - S000) / 3. + (S101 - S100) / 6. + (S011 - S010) / 6. + (S111 - S110) / 3.

        Rk = (R[0] * self.k0[0], R[1] * self.k0[1], R[2] * self.k0[2])
        (i0, j0, k0) = tuple(map(round, Rk))
        Rprimek = ((R[0] + dx) * self.k0[0], (R[1] + dy) * self.k0[1], (R[2] + dz) * self.k0[2])
        (i1, j1, k1) = tuple(map(round, Rprimek))

        (imin, imax) = (min(i0, i1), max(i0, i1))
        (jmin, jmax) = (min(j0, j1), max(j0, j1))
        (kmin, kmax) = (min(k0, k1), max(k0, k1))
        # deposit z current densities
        for i in range(0, self.N1):
            for j in range(0, self.N2):
                temp = 0
                C = -q * self.Dz / dt
                for k in range(kmin - 1, kmax + 2):
                    kred = k % self.N3
                    self.J[2][i][j][kred] += temp
                    temp += C * W3[i][j][kred]
        # deposit x current densities
        for j in range(0, self.N3):
            for k in range(0, self.N3):
                temp = 0
                C = -q * self.Dx / dt
                for i in range(imin - 1, imax + 2):
                    ired = i % self.N1
                    self.J[0][ired][j][k] += temp
                    temp += C * W1[ired][j][k]
        # deposit y current densities
        for k in range(0, self.N3):
            for i in range(0, self.N1):
                temp = 0
                C = -q * self.Dy / dt
                for j in range(jmin - 1, jmax + 2):
                    jred = j % self.N2
                    self.J[1][i][jred][k] += temp
                    temp += C * W2[i][jred][k]

    def set_semistatic_init_conds(self):
        # E = -gradm(phi), where lapl(phi) = -rho
        # this ensures that divp(E) = rho
        self.E = self.gradm(self.poisson_solve(self.rho))

        # B = curlm(A), where lapl(A) = -J
        # this ensures that curlp(B) ~~ J and divm(B) = 0
        self.B = -self.curlm(np.array([
            self.poisson_solve(self.J[0]),
            self.poisson_solve(self.J[1]),
            self.poisson_solve(self.J[2])
        ]))

    # TODO: replace np.fftn with np.rfftn (real Fourier transform), which is slightly faster
    def poisson_solve(self, source):
        # output f such that self.lapl(f) == source, and also the sum of f over all grid points is zero
        assert isinstance(source, np.ndarray) and source.shape == self.mesh_size

        rho_tilde = np.fft.fftn(source)  # compute the Fourier transform of charge density
        f_tilde = np.divide(rho_tilde, self.lapl_dft)
        f_tilde[0][0][0] = 0.  # set the constant term to zero
        return np.real(np.fft.ifftn(f_tilde))

    # TODO: test if these differential operators are implemented correctly, also perhaps optimize
    def pdp(self, i, f):
        # take the forward difference of a scalar function
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return self.k0[i] * (np.roll(f, -1, axis=i) - f)

    def pdm(self, i, f):
        # take the backward difference of a scalar function
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return self.k0[i] * (f - np.roll(f, 1, axis=i))

    def gradp(self, f):
        # the gradient operator, implemented with forward differences
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return np.array([
            self.pdp(0, f),
            self.pdp(1, f),
            self.pdp(2, f)
        ])

    def gradm(self, f):
        # the gradient operator, implemented with backward differences
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return np.array([
            self.pdm(0, f),
            self.pdm(1, f),
            self.pdm(2, f)
        ])

    def divp(self, v):
        # the divergence operator, implemented with forward differences
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        return self.pdp(0, v[0]) + self.pdp(1, v[1]) + self.pdp(2, v[2])

    def divm(self, v):
        # the divergence operator, implemented with backward differences
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        return self.pdm(0, v[0]) + self.pdm(1, v[1]) + self.pdm(2, v[2])

    def curlp(self, v):
        # the curl operator, implemented with forward differences
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        return np.array([
            self.pdp(1, v[2]) - self.pdp(2, v[1]),
            self.pdp(2, v[0]) - self.pdp(0, v[2]),
            self.pdp(0, v[1]) - self.pdp(1, v[0])
        ])

    def curlm(self, v):
        # the curl operator, implemented with backward differences
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        return np.array([
            self.pdm(1, v[2]) - self.pdm(2, v[1]),
            self.pdm(2, v[0]) - self.pdm(0, v[2]),
            self.pdm(0, v[1]) - self.pdm(1, v[0])
        ])

    def lapl(self, phi):
        # Laplacian operator
        assert isinstance(phi, np.ndarray) and phi.shape == self.mesh_size
        lapl = np.zeros(self.mesh_size)
        lapl += self.k0[0] ** 2 * (np.roll(phi, 1, axis=0)
                                   - 2 * phi
                                   + np.roll(phi, -1, axis=0))
        lapl += self.k0[1] ** 2 * (np.roll(phi, 1, axis=1)
                                   - 2 * phi
                                   + np.roll(phi, -1, axis=1))
        lapl += self.k0[2] ** 2 * (np.roll(phi, 1, axis=2)
                                   - 2 * phi
                                   + np.roll(phi, -1, axis=2))
        return lapl

    def evolve_fields(self, dt):
        # evolve the electric and magnetic field by one time step
        self.E += dt * (self.curlp(self.B) - self.J)
        self.B += dt * (-self.curlm(self.E))

    # NEED TO OPTIMIZE, AND UPDATE METHOD
    def get_EB_at(self, R):
        # interpolate electric and magnetic fields to a desired point (does not have to lie on the grid)
        assert isinstance(R, np.ndarray) and R.shape == (3,)

        Rk = (R[0] * self.k0[0], R[1] * self.k0[1], R[2] * self.k0[2])
        (i0, j0, k0) = tuple(map(round, Rk))

        S = self.compute_form_factor(R)
        return (self.DV * np.array([
            np.sum(np.multiply(self.E[0], S)),
            np.sum(np.multiply(self.E[1], S)),
            np.sum(np.multiply(self.E[2], S))
        ]),
        self.DV * np.array([
            np.sum(np.multiply(self.B[0], S)),
            np.sum(np.multiply(self.B[1], S)),
            np.sum(np.multiply(self.B[2], S))
        ]))


class Particle:
    def __init__(self, m, q, R, v):
        self.m = m
        self.q = q

        self.R = R
        self.v = v
        self.p = np.zeros(3)
        self.compute_p()

    def compute_p(self):
        self.p = self.v / np.sqrt(1 - self.v[0]**2 - self.v[1]**2 - self.v[2]**2)

    def compute_v(self):
        self.v = self.p / np.sqrt(1 + self.p[0]**2 + self.p[1]**2 + self.p[2]**2)

    # CHANGE PUSHER ALGORITHM
    def push(self, dt, E, B):
        self.p += (E + np.cross(self.v, B)) * self.q / self.m
        self.compute_v()
        self.R += dt * self.v


def test_1():
    # test if Grid.set_semistatic_init_conds maintains div E = rho and div B = 0
    N1 = random.randint(5, 200)
    N2 = random.randint(5, 200)
    N3 = random.randint(5, 200)
    (xsize, ysize, zsize) = 1. + 9. * np.random.rand(3)

    g = Grid((N1, N2, N3), (xsize, ysize, zsize))

    # randomize rho making sure that the total charge is zero
    g.rho = -1. + 2. * np.random.rand(N1, N2, N3)
    g.rho -= np.sum(g.rho) / (N1 * N2 * N3)

    # randomize J
    g.J = -1. + 2. * np.random.rand(3, N1, N2, N3)

    g.set_semistatic_init_conds()

    print('-- test 1 --')
    print(np.max(np.abs(g.divp(g.E) - g.rho)))  # div E - rho
    print(np.max(np.abs(g.divm(g.B))))          # div B
    print('----')


def main():
    test_1()

    '''particles = [
        Particle(1., 1., np.array([0.1, 0.2, 0.3]), np.array([-0.2, -0.2, 0.1])),
        Particle(1., -1., np.array([0.4, 0.5, 0.6]), np.array([0.9, 0.0, 0.0]))
    ]

    dt = 0.1

    g = Grid((10, 10, 10), (1., 1., 1.))

    # init conds
    g.refresh_charge()
    for p in particles:
        g.deposit_charge(p.q, p.R, p.v, dt)
    g.set_electrostatic_init_conds()

    # loop
    for i in range(20):
        qRvlist = []
        for p in particles:
            print(p.R)
            (E, B) = g.get_EB_at(p.R)
            p.push(dt, E, B)
            qRvlist.append((p.q, p.R, p.v))
        print('--')

        g.refresh_charge()
        for (q, R, v) in qRvlist:
            g.deposit_charge(q, R, v, dt)

        g.evolve_fields(dt)'''


if __name__ == '__main__':
    main()
