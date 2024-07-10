import numpy as np
import random

from FormFactor import FormFactor
from Quantum import q_back_diff, q_forw_diff


def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0

# ----------------------------------------------------------------------
#               WRAPAROUND ARRAY FUNCTIONS
# ----------------------------------------------------------------------
# The two functions wraparound_array_add and wraparound_array_dot are
# helper functions for working with torus arrays.


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

    d = A.ndim  # dimension of A and B
    assert A.ndim == B.ndim  # A and B must be of the same dimensions

    (sa, sb) = (A.shape, B.shape)
    assert all([sb[i] <= sa[i] for i in range(d)])  # B must be smaller than or same as A in every dimension

    # mod out ind by the shape of A
    ind = ind.copy()
    for i in range(d):
        ind[i] = ind[i] % sa[i]

    return wad_helper(A, B, ind, 0)


# -------------------------------------------------------------------------------
#                                  GRID
# -------------------------------------------------------------------------------
# A class to store and modify the electric and magnetic fields, as well as charge
# and current densities. Includes many complicated functions. The most complex
# functions are deposit_charge and get_EB_at.
#
# RELIES ON:
#   [*] FormFactor


class Grid:
    def __init__(self, mesh_size, phys_size, form_factor):
        assert isinstance(phys_size, tuple) and len(phys_size) == 3
        assert all(map(lambda x: isinstance(x, float), phys_size))

        assert isinstance(mesh_size, tuple) and len(mesh_size) == 3
        assert all(map(lambda x: isinstance(x, int), mesh_size))
        assert all(map(lambda x: x >= 2 * form_factor.g + 3, mesh_size))  # large enough for Esirkepov calculation

        assert isinstance(form_factor, FormFactor)

        self.phys_size = phys_size  # (xsize, ysize, zsize), say in meters
        self.mesh_size = mesh_size  # (N1, N2, N3), number of mesh points in each dimension

        self.form_factor = form_factor  # form factor of the super-particles

        (self.N1, self.N2, self.N3) = self.mesh_size                   # number of mesh points in each direction
        (self.Dx, self.Dy, self.Dz) = np.divide(phys_size, mesh_size)  # grid spacings
        self.DV = self.Dx * self.Dy * self.Dz                          # grid cell volume

        # the largest time step with which field evolution is stable (see the writeup)
        self.max_time_step = np.sqrt(1 / (1 / self.Dx**2 + 1 / self.Dy**2 + 1 / self.Dz**2))

        # (1/dx, 1/dy, 1/dz)
        self.k0 = np.divide(mesh_size, phys_size)
        self.one_over_dV = self.k0[0] * self.k0[1] * self.k0[2]

        # store the Fourier space representation of the lalplacian operator
        self.lapl_dft = np.zeros(self.mesh_size)
        self.initialize_lapl_dft()

        self.E = np.zeros((3,) + mesh_size)  # electric field
        self.B = np.zeros((3,) + mesh_size)  # magnetic field

        self.Ec = np.zeros((3,) + mesh_size)  # centered electric field, to be computed from self.E
        self.Bc = np.zeros((3,) + mesh_size)  # centered magnetic field, to be computed from self.B
        self.EcBc_initialized = False         # whether Ec and Bc has been computed

        self.rho = np.zeros(mesh_size)       # charge density
        self.J = np.zeros((3,) + mesh_size)  # current density

    # initialize lapl_dft, which encodes how the
    # discrete Fourier transform of a function relates to the original DFT
    def initialize_lapl_dft(self):
        arr1 = -4 / (self.Dx ** 2) * (np.sin(np.pi / self.N1 * np.array(range(0, self.N1))) ** 2)
        arr2 = -4 / (self.Dy ** 2) * (np.sin(np.pi / self.N2 * np.array(range(0, self.N2))) ** 2)
        arr3 = -4 / (self.Dz ** 2) * (np.sin(np.pi / self.N3 * np.array(range(0, self.N3))) ** 2)
        self.lapl_dft = (arr1[:, np.newaxis, np.newaxis] +
                         arr2[np.newaxis, :, np.newaxis] +
                         arr3[np.newaxis, np.newaxis, :])
        self.lapl_dft[0, 0, 0] = 1.  # set (0, 0, 0) component to 1. because normally it is 0. (see poisson_solve)

    # cross product of two vector fields
    def cross(self, A, B):
        assert isinstance(A, np.ndarray) and A.shape == (3,) + self.mesh_size
        assert isinstance(B, np.ndarray) and B.shape == (3,) + self.mesh_size

        return np.array([
            A[1] * B[2] - A[2] * B[1],
            A[2] * B[0] - A[0] * B[2],
            A[0] * B[1] - A[1] * B[0]
        ])

    # shift a scalar field by one grid cell in the positive i direction
    def shiftp(self, i, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return np.roll(f, -1, axis=i)

    # shift a scalar field by one grid cell in the negative i direction
    def shiftm(self, i, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return np.roll(f, 1, axis=i)

    # half-shift a scalar field by one grid cell in the positive i direction
    def half_shiftp(self, i, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return 0.5 * (np.roll(f, -1, axis=i) + f)

    # half-shift a scalar field by one grid cell in the negative i direction
    def half_shiftm(self, i, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return 0.5 * (f + np.roll(f, 1, axis=i))

    # centered cross product of two vector fields (see the writeup)
    def ccp(self, A, B):
        assert isinstance(A, np.ndarray) and A.shape == (3,) + self.mesh_size
        assert isinstance(B, np.ndarray) and B.shape == (3,) + self.mesh_size

        return np.array([
            A[1] * self.half_shiftp(0, B[2]) - A[2] * self.half_shiftp(0, B[1]),
            A[2] * self.half_shiftp(1, B[0]) - A[0] * self.half_shiftp(1, B[2]),
            A[0] * self.half_shiftp(2, B[1]) - A[1] * self.half_shiftp(2, B[0])
        ])

    # forward difference
    def pdp(self, i, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return self.k0[i] * (np.roll(f, -1, axis=i) - f)

    # backward difference
    def pdm(self, i, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return self.k0[i] * (f - np.roll(f, 1, axis=i))

    # forward difference computed using a quantum circuit
    def q_pdp(self, i, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        assert all(map(is_power_of_2, self.mesh_size))  # the grid dimensions must all be powers of 2

        if i == 0:
            result = np.zeros(self.mesh_size)
            for y in range(self.mesh_size[1]):
                for z in range(self.mesh_size[2]):
                    result[:, y, z] = self.k0[i] * q_forw_diff(f[:, y, z])
            return result
        elif i == 1:
            result = np.zeros(self.mesh_size)
            for z in range(self.mesh_size[2]):
                for x in range(self.mesh_size[0]):
                    result[x, :, z] = self.k0[i] * q_forw_diff(f[x, :, z])
            return result
        elif i == 2:
            result = np.zeros(self.mesh_size)
            for x in range(self.mesh_size[0]):
                for y in range(self.mesh_size[1]):
                    result[x, y, :] = self.k0[i] * q_forw_diff(f[x, y, :])
            return result

    # backward difference computed using a quantum circuit
    def q_pdm(self, i, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        assert all(map(is_power_of_2, self.mesh_size))  # the grid dimensions must all be powers of 2

        if i == 0:
            result = np.zeros(self.mesh_size)
            for y in range(self.mesh_size[1]):
                for z in range(self.mesh_size[2]):
                    result[:, y, z] = self.k0[i] * q_back_diff(f[:, y, z])
            return result
        elif i == 1:
            result = np.zeros(self.mesh_size)
            for z in range(self.mesh_size[2]):
                for x in range(self.mesh_size[0]):
                    result[x, :, z] = self.k0[i] * q_back_diff(f[x, :, z])
            return result
        elif i == 2:
            result = np.zeros(self.mesh_size)
            for x in range(self.mesh_size[0]):
                for y in range(self.mesh_size[1]):
                    result[x, y, :] = self.k0[i] * q_back_diff(f[x, y, :])
            return result

    # gradient with forward differences
    def gradp(self, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return np.array([
            self.pdp(0, f),
            self.pdp(1, f),
            self.pdp(2, f)
        ])

    # gradient with backward differences
    def gradm(self, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        return np.array([
            self.pdm(0, f),
            self.pdm(1, f),
            self.pdm(2, f)
        ])

    # gradient with forward differences computed using a quantum circuit
    def q_gradp(self, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        assert all(map(is_power_of_2, self.mesh_size))  # the grid dimensions must all be powers of 2

        return np.array([
            self.q_pdp(0, f),
            self.q_pdp(1, f),
            self.q_pdp(2, f)
        ])

    # gradient with backward differences computed using a quantum circuit
    def q_gradm(self, f):
        assert isinstance(f, np.ndarray) and f.shape == self.mesh_size
        assert all(map(is_power_of_2, self.mesh_size))  # the grid dimensions must all be powers of 2

        return np.array([
            self.q_pdm(0, f),
            self.q_pdm(1, f),
            self.q_pdm(2, f)
        ])

    # divergence with forward differences
    def divp(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        return self.pdp(0, v[0]) + self.pdp(1, v[1]) + self.pdp(2, v[2])

    # divergence with backward differences
    def divm(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        return self.pdm(0, v[0]) + self.pdm(1, v[1]) + self.pdm(2, v[2])

    # divergence with forward differences computed using a quantum circuit
    def q_divp(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        assert all(map(is_power_of_2, self.mesh_size))  # the grid dimensions must all be powers of 2

        return self.q_pdp(0, v[0]) + self.q_pdp(1, v[1]) + self.q_pdp(2, v[2])

    # divergence with backward differences computed using a quantum circuit
    def q_divm(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        assert all(map(is_power_of_2, self.mesh_size))  # the grid dimensions must all be powers of 2

        return self.q_pdm(0, v[0]) + self.q_pdm(1, v[1]) + self.q_pdm(2, v[2])

    # curl with forward differences
    def curlp(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        return np.array([
            self.pdp(1, v[2]) - self.pdp(2, v[1]),
            self.pdp(2, v[0]) - self.pdp(0, v[2]),
            self.pdp(0, v[1]) - self.pdp(1, v[0])
        ])

    # curl with backward differences
    def curlm(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        return np.array([
            self.pdm(1, v[2]) - self.pdm(2, v[1]),
            self.pdm(2, v[0]) - self.pdm(0, v[2]),
            self.pdm(0, v[1]) - self.pdm(1, v[0])
        ])

    # curl with forward differences computed using a quantum circuit
    def q_curlp(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        assert all(map(is_power_of_2, self.mesh_size))  # the grid dimensions must all be powers of 2
        return np.array([
            self.q_pdp(1, v[2]) - self.q_pdp(2, v[1]),
            self.q_pdp(2, v[0]) - self.q_pdp(0, v[2]),
            self.q_pdp(0, v[1]) - self.q_pdp(1, v[0])
        ])

    # curl with backward differences computed using a quantum circuit
    def q_curlm(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size
        assert all(map(is_power_of_2, self.mesh_size))  # the grid dimensions must all be powers of 2
        return np.array([
            self.q_pdm(1, v[2]) - self.q_pdm(2, v[1]),
            self.q_pdm(2, v[0]) - self.q_pdm(0, v[2]),
            self.q_pdm(0, v[1]) - self.q_pdm(1, v[0])
        ])

    # laplacian (centered)
    def lapl(self, phi):
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

    # laplacian (centered) computed using a quantum circuit
    def q_lapl(self, phi):
        assert all(map(is_power_of_2, self.mesh_size))  # the grid dimensions must all be powers of 2
        return self.q_divp(self.q_gradm(phi))

    # TODO: replace np.fftn with np.rfftn (real Fourier transform), which is slightly faster
    # output f such that self.lapl(f) == source, and also the sum of f over all grid points is zero
    # if np.sum(source) != 0, IGNORE the constant background
    def poisson_solve(self, source):
        assert isinstance(source, np.ndarray) and source.shape == self.mesh_size

        rho_tilde = np.fft.fftn(source)  # compute the Fourier transform of charge density
        f_tilde = np.divide(rho_tilde, self.lapl_dft)
        f_tilde[0, 0, 0] = 0.  # set the constant term to zero
        return np.real(np.fft.ifftn(f_tilde))

    # given a vector field v, subtract from it the backward gradient of a certain function to make divp v = 0
    def make_divp_zero(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size

        source = self.divp(v)
        alpha = self.poisson_solve(source)

        return v - self.gradm(alpha)

    # given a vector field v, subtract from it the forward gradient of a certain function to make divm v = 0
    def make_divm_zero(self, v):
        assert isinstance(v, np.ndarray) and v.shape == (3,) + self.mesh_size

        source = self.divm(v)
        alpha = self.poisson_solve(source)

        return v - self.gradp(alpha)

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
        g = self.form_factor.g                                    # support radius of the form factor
        S = np.zeros((2, 2, 2, 2 * g + 3, 2 * g + 3, 2 * g + 3))  # form factor array
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

        self.EcBc_initialized = False  # now that E and B have changed, Ec and Bc have to be recomputed

    # evolve the electric and magnetic field by one time step with differentiation done using a quantum circuit
    def q_evolve_fields(self, dt):
        assert 0 < dt <= self.max_time_step

        scale = 999.
        self.E += dt * (scale * self.q_curlp(self.B / scale) - self.J)
        self.B += dt * (-scale * self.q_curlm(self.E / scale))

        self.EcBc_initialized = False  # now that E and B have changed, Ec and Bc have to be recomputed

    # compute the centered version
    def compute_EB_centered(self):
        # compute the centered electric field
        self.Ec[0] = self.half_shiftp(0, self.E[0])
        self.Ec[1] = self.half_shiftp(1, self.E[1])
        self.Ec[2] = self.half_shiftp(2, self.E[2])

        # compute the centered magnetic field
        self.Bc[0] = self.half_shiftp(1, self.half_shiftp(2, self.B[0]))
        self.Bc[1] = self.half_shiftp(2, self.half_shiftp(0, self.B[1]))
        self.Bc[2] = self.half_shiftp(0, self.half_shiftp(1, self.B[2]))

        self.EcBc_initialized = True

    # interpolate electric and magnetic fields to a desired point (does not have to lie on the grid)
    # using the form factor of the super-particles
    def get_EB_at(self, R):
        assert isinstance(R, np.ndarray) and R.shape == (3,)
        assert self.EcBc_initialized  # need to call compute_EB_centered beforehand

        (xk, yk, zk) = R * self.k0 - 1/2  # subtract 1/2 because charge density is defined in the middle of a cell
        (i, j, k) = tuple(map(round, (xk, yk, zk)))
        (disp_x, disp_y, disp_z) = (xk - i, yk - j, zk - k)

        # compute the form factor array
        S = self.form_factor.get_array(disp_x, disp_y, disp_z)

        g = self.form_factor.g  # support radius of the super-particle form factor
        return (np.array([  # dot product of the centered electric field with the form factor
            wraparound_array_dot(self.Ec[0], S, [i - g, j - g, k - g]),
            wraparound_array_dot(self.Ec[1], S, [i - g, j - g, k - g]),
            wraparound_array_dot(self.Ec[2], S, [i - g, j - g, k - g])
        ]),
        np.array([  # dot product of the centered magnetic field with the form factor
            wraparound_array_dot(self.Bc[0], S, [i - g, j - g, k - g]),
            wraparound_array_dot(self.Bc[1], S, [i - g, j - g, k - g]),
            wraparound_array_dot(self.Bc[2], S, [i - g, j - g, k - g])
        ]))

    # compute the magnetic component of the Lorentz force acting on a particle of charge q
    # as it moves from position R0 at time t0 to position R1 at time t1 = t0 + dt
    # in magnetic field B0, which is at time t0
    def get_magnetic_force(self, B0, q, R0, R1, dt):
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
        g = self.form_factor.g                                    # support radius of the form factor
        S = np.zeros((2, 2, 2, 2 * g + 3, 2 * g + 3, 2 * g + 3))  # form factor array
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

        #J = np.zeros((3,) + self.mesh_size)
        #wraparound_array_add(J[0], Jx, [i0 - g, j0 - g - 1, k0 - g - 1])
        #wraparound_array_add(J[1], Jy, [i0 - g - 1, j0 - g, k0 - g - 1])
        #wraparound_array_add(J[2], Jz, [i0 - g - 1, j0 - g - 1, k0 - g])
        #print(J[0, 2])

        #return self.DV * np.sum(self.ccp(J, B0), axis=(1,2,3))

        # compute and return the Lorentz force
        return self.DV * np.array([
            wraparound_array_dot(self.half_shiftp(0, B0[2]), Jy, [i0 - g - 1, j0 - g, k0 - g - 1]) -
            wraparound_array_dot(self.half_shiftp(0, B0[1]), Jz, [i0 - g - 1, j0 - g - 1, k0 - g]),

            wraparound_array_dot(self.half_shiftp(1, B0[0]), Jz, [i0 - g - 1, j0 - g - 1, k0 - g]) -
            wraparound_array_dot(self.half_shiftp(1, B0[2]), Jx, [i0 - g, j0 - g - 1, k0 - g - 1]),

            wraparound_array_dot(self.half_shiftp(2, B0[1]), Jx, [i0 - g, j0 - g - 1, k0 - g - 1]) -
            wraparound_array_dot(self.half_shiftp(2, B0[0]), Jy, [i0 - g - 1, j0 - g, k0 - g - 1])
        ])
