import numpy as np
import random


# ----------------------------------------------------------------------
#               WRAPAROUND ARRAY FUNCTIONS
# ----------------------------------------------------------------------

# helper function to be called ONLY from wraparound_array_add
def waa_helper(A, B, ind, k):
    # k is the number of dimensions already checked
    if k == A.ndim:
        sl = tuple([slice(ind[i], ind[i]+B.shape[i]) for i in range(A.ndim)])
        A[sl] += B
    else:
        # check for overflow along the k-th dimension
        if ind[k] + B.shape[k] > A.shape[k]:
            # split B in two add the two parts to A separately
            sl1 = [slice(None) for i in range(A.ndim)]
            sl2 = sl1.copy()

            sl1[k] = slice(None, A.shape[k] - ind[k])
            sl2[k] = slice(A.shape[k] - ind[k], None)

            ind2 = ind.copy()
            ind2[k] = 0

            waa_helper(A, B[tuple(sl1)], ind, k+1)
            waa_helper(A, B[tuple(sl2)], ind2, k+1)
        else:
            # this dimension cleared, proceed to the next one
            waa_helper(A, B, ind, k+1)


# given two np arrays A, B both of dimension d and where A is larger than or same as B among every dimension,
# and an index ind = (i_0, ..., i_{d-1}),
# if ind = (0, ..., 0), then simply increment the bottom left part of A by B,
# and if ind is close to the upper right corner of A, add B to A wrapping around
def wraparound_array_add(A, B, ind):
    assert isinstance(ind, list) and all(map(lambda x: isinstance(x, int), ind))
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray)

    # check that B is smaller than or same size as A in every dimension
    (sa, sb) = (A.shape, B.shape)
    d = len(sa)  # dimension of A and B
    assert len(sa) == len(sb) and all([sb[i] <= sa[i] for i in range(d)])

    # mod out ind by the shape of A
    ind = ind.copy()
    for i in range(d):
        ind[i] = ind[i] % sa[i]

    waa_helper(A, B, ind, 0)


# helper function to be called ONLY from wraparound_array_dot
def wad_helper(A, B, ind, k):
    # k is the number of dimensions already checked
    if k == A.ndim:
        sl = tuple([slice(ind[i], ind[i]+B.shape[i]) for i in range(A.ndim)])
        return np.sum(np.multiply(A[sl], B))
    else:
        # check for overflow along the k-th dimension
        if ind[k] + B.shape[k] > A.shape[k]:
            # split B in two add the two parts to A separately
            sl1 = [slice(None) for i in range(A.ndim)]
            sl2 = sl1.copy()

            sl1[k] = slice(None, A.shape[k] - ind[k])
            sl2[k] = slice(A.shape[k] - ind[k], None)

            ind2 = ind.copy()
            ind2[k] = 0

            return wad_helper(A, B[tuple(sl1)], ind, k+1) + wad_helper(A, B[tuple(sl2)], ind2, k+1)
        else:
            # this dimension cleared, proceed to the next one
            return wad_helper(A, B, ind, k+1)


# given two np arrays A, B both of dimension d and where A is larger than or same as B among every dimension,
# and an index ind = (i_0, ..., i_{d-1}),
# if ind = (0, ..., 0), then compute np.sum(np.multiply(A[sl], B)), where sl picks out the left bottom part of A,
# and if ind is close to the upper right corner of A, dot B with A wrapping around
def wraparound_array_dot(A, B, ind):
    assert isinstance(ind, list) and all(map(lambda x: isinstance(x, int), ind))
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray)

    # check that B is smaller than or same size as A in every dimension
    (sa, sb) = (A.shape, B.shape)
    d = len(sa)  # dimension of A and B
    assert len(sa) == len(sb) and all([sb[i] <= sa[i] for i in range(d)])

    # mod out ind by the shape of A
    ind = ind.copy()
    for i in range(d):
        ind[i] = ind[i] % sa[i]

    return wad_helper(A, B, ind, 0)


# -------------------------------------------------------------------------------------
#                                 FORM FACTOR
# -------------------------------------------------------------------------------------

class FormFactor:
    def __init__(self, func, g):
        # form factor whose value at a point (x, y, z) is func(x, y, z)
        # and whose support is contained in [-g, g]^3
        assert isinstance(g, int) and g >= 0

        self.func = func  # form factor function
        self.g = g        # support radius

        # compute the Cartesian product [-g, g]^3, to be used in get_array
        (self.i_grid, self.j_grid, self.k_grid) = np.indices((2 * g + 1, 2 * g + 1, 2 * g + 1)) - g

    # return an array containing self.func(p - disp) for p in the cube [-g, g]^3
    def get_array(self, disp_x, disp_y, disp_z):
        assert isinstance(disp_x, float) and -0.5 <= disp_x <= 0.5
        assert isinstance(disp_y, float) and -0.5 <= disp_y <= 0.5
        assert isinstance(disp_z, float) and -0.5 <= disp_z <= 0.5

        # CREDIT: to implement this using numpy functions instead of Python loops, I used ChatGPT 4o
        return self.func(self.i_grid - disp_x, self.j_grid - disp_y, self.k_grid - disp_z)


# the one-dimensional quadratic spline function, vectorized for numpy use
def one_dim_quadr_spline(x):
    # @pre: -3/2 <= x <= 3/2
    # CREDIT: to vectorize the function, I used ChatGPT 4o
    x = np.asarray(x)  # ensure x is a numpy array

    # calculate the spline values using vectorized operations
    return np.where(
        x < -0.5,
        0.5 * (1.5 + x) ** 2,
        np.where(
            x < 0.5,
            0.75 - x ** 2,
            0.5 * (1.5 - x) ** 2
        )
    )


QuadraticSplineFF = FormFactor(lambda x, y, z: one_dim_quadr_spline(x) *
                                               one_dim_quadr_spline(y) *
                                               one_dim_quadr_spline(z), 1)


# ------------------------------------------------------------------------------
#                                 PARTICLE
# ------------------------------------------------------------------------------

class Particle:
    def __init__(self, m, q, R, v=None, p=None):
        self.m = m
        self.q = q

        self.R = R

        if not isinstance(v, type(None)):
            self.v = v
            self.p = np.zeros(3)
            self.compute_p()
        elif not isinstance(p, type(None)):
            self.p = p
            self.v = np.zeros(3)
            self.compute_v()
        else:
            self.v = np.zeros(3)
            self.p = np.zeros(3)

    def compute_p(self):
        self.p = self.v / np.sqrt(1 - self.v[0]**2 - self.v[1]**2 - self.v[2]**2)

    def compute_v(self):
        self.v = self.p / np.sqrt(1 + self.p[0]**2 + self.p[1]**2 + self.p[2]**2)

    def push_init(self, dt):
        self.R += dt * self.v

    def push(self, dt, E, B):
        self.p += (E + np.cross(self.v, B)) * self.q / self.m
        self.compute_v()

        self.R += dt * self.v


# create a randomized particle
def random_particle(R_range, p_range=10., q_range=1., m_range=(0.1, 1.)):
    # generate a particle whose location is chosen at random from [0, R_range[0]] cross ... cross [0, R_range[2]]
    # whose momentum is chosen at random from [-p_range, p_range]^3,
    # and whose charge and mass are chosen at random from [-q_range, q_range] and [m_range[0], m_range[1]]
    assert isinstance(R_range, tuple) and len(R_range) == 3 and all(map(lambda x: x > 0, R_range))
    assert isinstance(p_range, float) and p_range > 0
    assert isinstance(q_range, float) and q_range > 0
    (m_min, m_max) = m_range
    assert isinstance(m_min, float) and isinstance(m_max, float) and 0 < m_min < m_max

    R = np.array(R_range) * np.random.rand(3)
    p = p_range * (-1. + 2. * np.random.rand(3))
    q = q_range * (-1. + 2. * random.random())
    m = m_min + (m_max - m_min) * random.random()

    return Particle(m, q, R, p=p)


# -------------------------------------------------------------------------------
#                                  GRID
# -------------------------------------------------------------------------------

class Grid:
    def __init__(self, mesh_size, phys_size, form_factor):
        assert isinstance(phys_size, tuple) and len(phys_size) == 3
        assert all(map(lambda x: isinstance(x, float), phys_size))

        assert isinstance(mesh_size, tuple) and len(mesh_size) == 3
        assert all(map(lambda x: isinstance(x, int) and x > 0, mesh_size))

        assert isinstance(form_factor, FormFactor)

        self.phys_size = phys_size  # (xsize, ysize, zsize), say in meters
        self.mesh_size = mesh_size  # (N_1, N_2, N_3), number of mesh points in each dimension

        self.form_factor = form_factor

        (self.N1, self.N2, self.N3) = self.mesh_size
        (self.Dx, self.Dy, self.Dz) = np.divide(phys_size, mesh_size)
        self.DV = self.Dx * self.Dy * self.Dz

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

    # TODO: replace np.fftn with np.rfftn (real Fourier transform), which is slightly faster
    # output f such that self.lapl(f) == source, and also the sum of f over all grid points is zero
    # if np.sum(source) != 0, IGNORE the constant background
    def poisson_solve(self, source):
        assert isinstance(source, np.ndarray) and source.shape == self.mesh_size

        rho_tilde = np.fft.fftn(source)  # compute the Fourier transform of charge density
        f_tilde = np.divide(rho_tilde, self.lapl_dft)
        f_tilde[0][0][0] = 0.  # set the constant term to zero
        return np.real(np.fft.ifftn(f_tilde))

    # set rho and J to zero
    def refresh_charge(self):
        self.rho.fill(0)
        self.J.fill(0)

    # calculate the charge and current densities induced by a particle of charge q
    # moving from position R0 at time t_0 over to position R1 at time t_1 = t_0 + dt,
    # and rho is at time t_1 and J is at time t_0
    def deposit_charge(self, q, R0, R1, dt):
        assert isinstance(R0, np.ndarray) and R0.shape == (3,)
        assert isinstance(R1, np.ndarray) and R1.shape == (3,)
        assert abs(R1[0] - R0[0]) <= self.Dx and abs(R1[1] - R0[1]) <= self.Dy and abs(R1[2] - R0[2]) <= self.Dz
        assert np.linalg.norm(R1 - R0) < dt  # going slower than light
        assert dt > 0

        # initial position, divided by grid cell size;
        # the closest grid point; and
        # the difference between the position and the closest grid point, respectively
        (x0k, y0k, z0k) = R0 * self.k0
        (i0, j0, k0) = tuple(map(round, (x0k, y0k, z0k)))
        (disp_x0, disp_y0, disp_z0) = (x0k - i0, y0k - j0, z0k - k0)

        # same, but for final position
        (x1k, y1k, z1k) = R1 * self.k0
        (i1, j1, k1) = tuple(map(round, (x1k, y1k, z1k)))
        (disp_x1, disp_y1, disp_z1) = (x1k - i1, y1k - j1, z1k - k1)

        # the change in the location of the nearest grid cell
        (Di, Dj, Dk) = (i1 - i0, j1 - j0, k1 - k0)

        # compute the form factors of the particle over eight different positions
        g = self.form_factor.g
        S = np.zeros((2, 2, 2, 2 * g + 3, 2 * g + 3, 2 * g + 3))
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    S_raw = self.form_factor.get_array(
                        disp_x0 if a == 0 else disp_x1,
                        disp_y0 if b == 0 else disp_y1,
                        disp_z0 if c == 0 else disp_z1
                    )  # np array of dimension (2 * g + 1)^3
                    pos_x = 1 if a == 0 else 1 + Di
                    pos_y = 1 if b == 0 else 1 + Dj
                    pos_z = 1 if c == 0 else 1 + Dk
                    S[a, b, c, pos_x:pos_x+2*g+1, pos_y:pos_y+2*g+1, pos_z:pos_z+2*g+1] += S_raw

        # compute Esirkepov's W vector
        Wx = ((S[1, 0, 0] - S[0, 0, 0]) / 3. + (S[1, 1, 0] - S[0, 1, 0]) / 6.
              + (S[1, 0, 1] - S[0, 0, 1]) / 6. + (S[1, 1, 1] - S[0, 1, 1]) / 3.)
        Wy = ((S[0, 1, 0] - S[0, 0, 0]) / 3. + (S[1, 1, 0] - S[1, 0, 0]) / 6.
              + (S[0, 1, 1] - S[0, 0, 1]) / 6. + (S[1, 1, 1] - S[1, 0, 1]) / 3.)
        Wz = ((S[0, 0, 1] - S[0, 0, 0]) / 3. + (S[1, 0, 1] - S[1, 0, 0]) / 6.
              + (S[0, 1, 1] - S[0, 1, 0]) / 6. + (S[1, 1, 1] - S[1, 1, 0]) / 3.)

        # compute the currents induced by the motion of the particle
        Jx = -q * self.one_over_dV * self.Dx / dt * np.cumsum(Wx, axis=0)
        Jy = -q * self.one_over_dV * self.Dy / dt * np.cumsum(Wy, axis=1)
        Jz = -q * self.one_over_dV * self.Dz / dt * np.cumsum(Wz, axis=2)

        # add the computed currents, placed appropriately, to current densities
        wraparound_array_add(self.J[0], Jx, [i0 - g, j0 - g - 1, k0 - g - 1])
        wraparound_array_add(self.J[1], Jy, [i0 - g - 1, j0 - g, k0 - g - 1])
        wraparound_array_add(self.J[2], Jz, [i0 - g - 1, j0 - g - 1, k0 - g])

        # add S111, with appropriate factors and placed appropriately, to charge density
        wraparound_array_add(self.rho, q * self.one_over_dV * S[1, 1, 1], [i0 - g - 1, j0 - g - 1, k0 - g - 1])

    # set E and B to be "almost stationary" (see the writeup)
    def set_semistatic_init_conds(self):
        # check if the total charge is (approximately) zero
        assert abs(self.DV * np.sum(self.rho)) < 0.1**9

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

    # evolve the electric and magnetic field by one time step
    def evolve_fields(self, dt):
        self.E += dt * (self.curlp(self.B) - self.J)
        self.B += dt * (-self.curlm(self.E))

    # interpolate electric and magnetic fields to a desired point (does not have to lie on the grid)
    # using the form factor of the super-particles
    # TODO: check if this works correctly, perhaps improve interpolation method
    def get_EB_at(self, R):
        assert isinstance(R, np.ndarray) and R.shape == (3,)

        (xk, yk, zk) = R * self.k0
        (i, j, k) = tuple(map(round, (xk, yk, zk)))
        (disp_x, disp_y, disp_z) = (xk - i, yk - j, zk - k)

        S = self.form_factor.get_array(disp_x, disp_y, disp_z)
        g = self.form_factor.g
        return (np.array([
            wraparound_array_dot(self.E[0], S, [i - g - 1, j - g - 1, k - g - 1]),
            wraparound_array_dot(self.E[1], S, [i - g - 1, j - g - 1, k - g - 1]),
            wraparound_array_dot(self.E[2], S, [i - g - 1, j - g - 1, k - g - 1])
        ]),
        np.array([
            wraparound_array_dot(self.B[0], S, [i - g - 1, j - g - 1, k - g - 1]),
            wraparound_array_dot(self.B[1], S, [i - g - 1, j - g - 1, k - g - 1]),
            wraparound_array_dot(self.B[2], S, [i - g - 1, j - g - 1, k - g - 1])
        ]))


# ------------------------------------------------------------------------------
#                              TESTS AND MAIN
# ------------------------------------------------------------------------------

# test if charge deposition maintains d_dt rho + div J = 0
# Grid.deposit_charge, Particle.push_init
def test_0():
    # generate a random grid
    N1 = random.randint(5, 200)
    N2 = random.randint(5, 200)
    N3 = random.randint(5, 200)
    (xsize, ysize, zsize) = 1. + 9. * np.random.rand(3)

    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # choose a dt that would be safe
    dt = (0.01 + 0.98 * random.random()) * min(xsize / N1, ysize / N2, zsize / N3)

    # generate a collection of random particles
    nparticles = random.randint(1, 40)
    particles = [random_particle((xsize, ysize, zsize)) for i in range(nparticles)]

    # compute rho at time 1 and J at time 0,
    # with particles moving from R_0 to R_1
    g.refresh_charge()
    for ptc in particles:
        g.deposit_charge(ptc.q, ptc.R, ptc.R + ptc.v * dt, dt)
    rho_1 = g.rho.copy()

    # move particles from R_0 to R_1
    # (IMPORTANT) for the initial time steps, the particles must be pushed WITHOUT modifying their velocities,
    # otherwise cannot guarantee charge conservation
    for ptc in particles:
        ptc.push_init(dt)

    # compute rho at time 2 and J at time 1,
    # with particles moving from R_1 to R_2
    g.refresh_charge()
    for ptc in particles:
        g.deposit_charge(ptc.q, ptc.R, ptc.R + ptc.v * dt, dt)
    rho_2 = g.rho.copy()

    # compute the error
    e = (rho_2 - rho_1) / dt + g.divp(g.J)
    print('-- test 0 --')
    print('mp', np.max(np.abs(e)))  # pd^+_0 rho + div^+ cdot J = 0 should hold to machine precision
    print('----')


# test if initial conditions satisfy div E = rho and div B = 0
# Grid.set_semistatic_init_conds, Grid.divp, Grid.divm
def test_1():
    # generate a random grid
    N1 = random.randint(5, 200)
    N2 = random.randint(5, 200)
    N3 = random.randint(5, 200)
    (xsize, ysize, zsize) = 1. + 9. * np.random.rand(3)

    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # randomize rho making sure that the total charge is zero
    g.rho = -1. + 2. * np.random.rand(N1, N2, N3)
    g.rho -= np.sum(g.rho) / (N1 * N2 * N3)

    # randomize J
    g.J = -1. + 2. * np.random.rand(3, N1, N2, N3)

    # compute initial fields
    g.set_semistatic_init_conds()

    # compute the error
    print('-- test 1 --')
    print('mp', np.max(np.abs(g.divp(g.E) - g.rho)))  # div E - rho = 0 should hold to machine precision
    print('mp', np.max(np.abs(g.divm(g.B))))          # div B = 0 should hold to machine precision
    print('----')


# test if div E = rho and div B = 0 are maintained through time evolution
# Grid.set_semistatic_init_conds, Grid.evolve_fields, Grid.deposit_charge, Particle.push
def test_2():
    # generate a random grid
    N1 = random.randint(5, 200)
    N2 = random.randint(5, 200)
    N3 = random.randint(5, 200)
    (xsize, ysize, zsize) = 1. + 9. * np.random.rand(3)

    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # generate some random particles, ensuring that the total charge is zero
    nparticles = random.randint(0, 40)
    particles = [Particle(1. + 3. * random.random(),                            # mass
                          -1. + 2. * random.random(),                           # charge
                          np.array([xsize, ysize, zsize]) * np.random.rand(3),  # position
                          p=-1. + 2. * np.random.rand(3))                       # momentum
        for i in range(nparticles)]
    particles[-1].q = -sum(map(lambda ptc: ptc.q, particles[:-1]))

    # pick a time step which is not too large
    dt = (0.01 + 0.98 * random.random()) * min(xsize / N1, ysize / N2, zsize / N3)

    # set initial conditions
    g.refresh_charge()
    for ptc in particles:
        R0 = ptc.R.copy()
        ptc.push_init(dt)
        g.deposit_charge(ptc.q, R0, ptc.R, dt)
    g.set_semistatic_init_conds()

    # run the simulation for some number of time steps
    nsteps = random.randint(2, 30)
    for i in range(nsteps):
        # @pre: each particle is ptc = (q, R_t, v_t), and also g.rho is at time t and g.J at time t-1,
        # and g.E and g.B are at time t

        # change the particles' momentum and position, recording their previous position
        qR0R1list = []
        for ptc in particles:
            # Maxwell's equations constraints must be satisfied regardless of how the particles' momentum is changed,
            # so E and B can be picked at random
            E = -1. + 2. * np.random.rand(3)
            B = -1. + 2. * np.random.rand(3)

            # IMPORTANT: first update velocity, then use this updated velocity to modify position
            R0 = ptc.R.copy()
            ptc.push(dt, E, B)
            qR0R1list.append((ptc.q, R0, ptc.R))
            # at this point, ptc = (q, R_{t+1}, v_{t+1}),
            # where (IMPORTANT) R_{t+1} = R_t + dt * v_{t+1}, otherwise cannot guarantee charge conservation

        # update charge and current densities, as well as field strengths
        g.refresh_charge()
        for (q, R0, R1) in qR0R1list:
            g.deposit_charge(q, R0, R1, dt)
        # at this point, g.rho is at time t+1 because computed using particles' positions at time t+1,
        # an g.J is at time t

        g.evolve_fields(dt)
        # at this point, g.E and g.B are at time t+1

    # compute the errors
    print('-- test 2 --')
    print('mp', np.max(np.abs(g.divp(g.E) - g.rho)))  # div E - rho should hold to machine precision
    print('mp', np.max(np.abs(g.divm(g.B))))          # div B should hold to machine precision
    print('----')


# test if the field of two static charges is close to what we expect in the continuous case
def soft_test_0():
    g = Grid((200, 200, 200), (1., 1., 1.), QuadraticSplineFF)

    q = 1.
    x = 0.1
    particles = [
        Particle(1., q, np.array([-x, 0, 0])),
        Particle(1., -q, np.array([x, 0, 0]))
    ]  # a positive particle on the negative x axis and a negative particle on the positive x axis

    dt = 0.1

    # set initial conditions
    g.refresh_charge()
    for ptc in particles:
        R0 = ptc.R.copy()
        ptc.push_init(dt)
        g.deposit_charge(ptc.q, R0, ptc.R, dt)
    g.set_semistatic_init_conds()

    # compare result to expectation
    (E, B) = g.get_EB_at(np.zeros(3))
    print('-- soft test 0 --')
    print('Expect:')
    print('E =', np.array([2 * q / (4 * np.pi) / x**2, 0, 0]))
    print('B =', np.zeros(3))
    print('Find:')
    print('E =', E)
    print('B =', B)
    print('----')


# test if can reproduce the magnetic field of a current-carrying loop
def soft_test_1():
    g = Grid((200, 200, 200), (1., 1., 1.), QuadraticSplineFF)

    # a ring of total charge Q, with its center at (0, 0, -h) and having radius R,
    # rotating with angular velocity omega;
    # also there is an identical ring but with charge -Q and not rotating
    Q = 1.
    R = 0.2
    h = 0.12
    omega = 4.5

    # the number of
    N = 40
    q = Q / N

    particles = [
        Particle(1., q, np.array([R * np.cos(th), R * np.sin(th), -h]),
                        np.array([-R * omega * np.sin(th), R * omega * np.cos(th), 0]))
    for th in 2 * np.pi / N * np.arange(0, N)]
    particles += [
        Particle(1., -q, np.array([R * np.cos(th), R * np.sin(th), -h]),
                        np.zeros(3))
    for th in 2 * np.pi / N * np.arange(0, N)]

    dt = 0.0001

    # set initial conditions
    g.refresh_charge()
    for ptc in particles:
        R0 = ptc.R.copy()
        ptc.push_init(dt)
        g.deposit_charge(ptc.q, R0, ptc.R, dt)
    g.set_semistatic_init_conds()

    # compare result to expectation
    (E, B) = g.get_EB_at(np.zeros(3))
    print('-- soft test 1 --')
    print('Expect:')
    print('E =', np.zeros(3))
    print('B =', np.array([0, 0, Q * omega / (4 * np.pi) * R**2 / np.sqrt(R**2 + h**2)**3]))
    print('Find:')
    print('E =', E)
    print('B =', B)
    print('----')


def main():
    np.set_printoptions(suppress=True, precision=2)
    test_0()
    test_1()
    test_2()

    soft_test_0()
    soft_test_1()

    input('Press ENTER to complete')


if __name__ == '__main__':
    main()
