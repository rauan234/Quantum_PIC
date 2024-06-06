import numpy as np
import random

from FormFactor import FormFactor

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

# -------------------------------------------------------------------------------
#                                  GRID
# -------------------------------------------------------------------------------


class Grid:
    def __init__(self, mesh_size, phys_size, form_factor):
        assert isinstance(phys_size, tuple) and len(phys_size) == 3
        assert all(map(lambda x: isinstance(x, float), phys_size))

        assert isinstance(mesh_size, tuple) and len(mesh_size) == 3
        assert all(map(lambda x: isinstance(x, int), mesh_size))
        assert all(map(lambda x: x >= 2 * form_factor.g + 3, mesh_size))  # large enough for Esirkepov calculation

        assert isinstance(form_factor, FormFactor)

        self.phys_size = phys_size  # (xsize, ysize, zsize), say in meters
        self.mesh_size = mesh_size  # (N_1, N_2, N_3), number of mesh points in each dimension

        self.form_factor = form_factor

        (self.N1, self.N2, self.N3) = self.mesh_size
        (self.Dx, self.Dy, self.Dz) = np.divide(phys_size, mesh_size)
        self.DV = self.Dx * self.Dy * self.Dz

        # the largest time step with which field evolution is stable (see the writeup)
        self.max_time_step = np.sqrt(1 / (1 / self.Dx**2 + 1 / self.Dy**2 + 1 / self.Dz**2))

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
        self.lapl_dft[0, 0, 0] = 1.  # set (0, 0, 0) component to 1. because normally it is 0. (see poisson_solve)

        self.E = np.zeros((3,) + mesh_size)
        self.B = np.zeros((3,) + mesh_size)

        self.rho = np.zeros(mesh_size)
        self.J = np.zeros((3,) + mesh_size)

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
        f_tilde[0, 0, 0] = 0.  # set the constant term to zero
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
        (x0k, y0k, z0k) = R0 * self.k0 - 1/2  # subtract 1/2 because charge density is defined in the middle of a cell
        (i0, j0, k0) = tuple(map(round, (x0k, y0k, z0k)))
        (disp_x0, disp_y0, disp_z0) = (x0k - i0, y0k - j0, z0k - k0)

        # same, but for final position
        (x1k, y1k, z1k) = R1 * self.k0 - 1/2  # subtract 1/2 because charge density is defined in the middle of a cell
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
        assert abs(self.DV * np.sum(self.rho)) < 1e-9  # the total charge must be zero

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
        assert 0 < dt <= self.max_time_step

        self.E += dt * (self.curlp(self.B) - self.J)
        self.B += dt * (-self.curlm(self.E))

    # interpolate electric and magnetic fields to a desired point (does not have to lie on the grid)
    # using the form factor of the super-particles
    def get_EB_at(self, R):
        assert isinstance(R, np.ndarray) and R.shape == (3,)

        (xk, yk, zk) = R * self.k0 - 1/2  # subtract 1/2 because charge density is defined in the middle of a cell
        (i, j, k) = tuple(map(round, (xk, yk, zk)))
        (disp_x, disp_y, disp_z) = (xk - i, yk - j, zk - k)

        # initialize the form factor array and modify it according to the way described in the writeup
        # to shift its balancing by (-1/2, -1/2, -1/2)
        S = self.form_factor.get_array(disp_x, disp_y, disp_z)
        S = np.pad(S, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)  # pad with zeros outside
        S = 0.5 * (S + np.roll(S, 1, axis=0))  # shift x centering by -1/2
        S = 0.5 * (S + np.roll(S, 1, axis=1))  # shift y centering by -1/2
        S = 0.5 * (S + np.roll(S, 1, axis=2))  # shift z centering by -1/2

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
