import matplotlib.pyplot as plt
import numpy as np
import random

from Simulation import *
from Grid import *
from Particle import *
from FormFactor import *


# ------------------------------------------------------------------------------
#                              TESTS
# ------------------------------------------------------------------------------

# test Grid.pdp and Grid.pdm on a simple example,
# in particular testing if the derivative correctly wraps around
# @test: Grid.pdp, Grid.pdm
def unit_test_0():
    print('-- unit test 0 --')

    (N1, N2, N3) = (4, 5, 6)
    (xsize, ysize, zsize) = (2., 3., 7.)
    (Dx, Dy, Dz) = (xsize / N1, ysize / N2, zsize / N3)

    # initialize an example function
    f = np.zeros((N1, N2, N3))
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                f[i, j, k] = (i + 3) * (j + 3) * (k + 3)

    # compute the derivatives of f
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), EmptyFF)
    (f0p, f0m) = (g.pdp(0, f), g.pdm(0, f))
    (f1p, f1m) = (g.pdp(1, f), g.pdm(1, f))
    (f2p, f2m) = (g.pdp(2, f), g.pdm(2, f))

    # compare the result with what is expected
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                # test pdp(0, f)
                expected = -(N1-1) * (j + 3) * (k + 3) / Dx if i == N1 - 1 else (j + 3) * (k + 3) / Dx
                assert abs(f0p[i, j, k] - expected) < 1e-10

                # test pdm(0, f)
                expected = -(N1-1) * (j + 3) * (k + 3) / Dx if i == 0 else (j + 3) * (k + 3) / Dx
                assert abs(f0m[i, j, k] - expected) < 1e-10

                # test pdp(1, f)
                expected = -(N2-1) * (i + 3) * (k + 3) / Dy if j == N2 - 1 else (i + 3) * (k + 3) / Dy
                assert abs(f1p[i, j, k] - expected) < 1e-10

                # test pdm(1, f)
                expected = -(N2-1) * (i + 3) * (k + 3) / Dy if j == 0 else (i + 3) * (k + 3) / Dy
                assert abs(f1m[i, j, k] - expected) < 1e-10

                # test pdp(2, f)
                expected = -(N3-1) * (i + 3) * (j + 3) / Dz if k == N3 - 1 else (i + 3) * (j + 3) / Dz
                assert abs(f2p[i, j, k] - expected) < 1e-10

                # test pdm(2, f)
                expected = -(N3-1) * (i + 3) * (j + 3) / Dz if k == 0 else (i + 3) * (j + 3) / Dz
                assert abs(f2m[i, j, k] - expected) < 1e-10


# test if the discretized differential operators relate to each other correctly, part 1: scalar fields
# @test: Grid.gradp, Grid.gradm, Grid.divp, Grid.divm, Grid.curlp, Grid.curlm, Grid.lapl
def unit_test_1():
    print('-- unit test 1 --')

    # initialize the grid with some parameters
    (N1, N2, N3) = (200, 5, 13)
    (xsize, ysize, zsize) = (2., 3., 7.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), EmptyFF)

    # initialize some random scalar field
    f = -1. + 2. * np.random.rand(N1, N2, N3)

    # check that the derivatives all commute with one another
    for i in range(3):
        for j in range(3):
            assert np.max(np.abs(
                g.pdp(i, g.pdp(j, f)) - g.pdp(j, g.pdp(i, f))
            )) < 1e-10
            assert np.max(np.abs(
                g.pdp(i, g.pdm(j, f)) - g.pdm(j, g.pdp(i, f))
            )) < 1e-10
            assert np.max(np.abs(
                g.pdm(i, g.pdm(j, f)) - g.pdm(j, g.pdm(i, f))
            )) < 1e-10

    # check that the gradient is the vector of derivatives
    assert np.max(np.abs(
        g.gradp(f) - np.array([g.pdp(0, f), g.pdp(1, f), g.pdp(2, f)])
    )) < 1e-10
    assert np.max(np.abs(
        g.gradm(f) - np.array([g.pdm(0, f), g.pdm(1, f), g.pdm(2, f)])
    )) < 1e-10

    # check that the curl of a gradient is zero
    assert np.max(np.abs(
        g.curlp(g.gradp(f))
    )) < 1e-10
    assert np.max(np.abs(
        g.curlm(g.gradm(f))
    )) < 1e-10

    # check that the Laplacian is the divergence of the gradient
    assert np.max(np.abs(
        g.lapl(f) - g.divp(g.gradm(f))
    )) < 1e-10
    assert np.max(np.abs(
        g.lapl(f) - g.divm(g.gradp(f))
    )) < 1e-10


# test if the discretized differential operators relate to each other correctly, part 2: vector fields
# @test: Grid.divp, Grid.divm, Grid.curlp, Grid.curlm
def unit_test_2():
    print('-- unit test 2 --')

    # initialize the grid with some parameters
    (N1, N2, N3) = (200, 5, 13)
    (xsize, ysize, zsize) = (2., 3., 7.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), EmptyFF)

    # initialize some random vector field
    v = -1. + 2. * np.random.rand(3, N1, N2, N3)

    # check that the divergence of a curl is zero
    assert np.max(np.abs(g.divp(g.curlp(v)))) < 1e-10
    assert np.max(np.abs(g.divm(g.curlm(v)))) < 1e-10


# test if for a constant function, all differential operators are zero
# @test: Grid.gradp, Grid.gradm, Grid.divp, Grid.divm, Grid.curlp, Grid.curlm, Grid.lapl
def unit_test_3():
    print('-- unit test 3 --')

    # initialize the grid with some parameters
    (N1, N2, N3) = (200, 5, 13)
    (xsize, ysize, zsize) = (2., 3., 7.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), EmptyFF)

    # initialize constant scalar and vector fields
    f = np.zeros((N1, N2, N3))
    v = np.zeros((3, N1, N2, N3))
    f += -1. + 2. * np.random.rand(1)
    v += -1. + 2. * np.random.rand(1)

    # derivative operators
    for i in range(3):
        assert np.max(np.abs(g.pdp(i, f))) < 1e-10
        assert np.max(np.abs(g.pdm(i, f))) < 1e-10

    # scalar field vector operators
    for func in (g.gradp, g.gradm, g.lapl):
        assert np.max(np.abs(func(f))) < 1e-10

    # vector field vector operators
    for func in (g.divp, g.divm, g.curlp, g.curlm):
        assert np.max(np.abs(func(v))) < 1e-10


# test if the Poisson equation solver works correctly
# @test: Grid.poisson_solve
def unit_test_4():
    print('-- unit test 4 --')

    # initialize the grid with some parameters
    (N1, N2, N3) = (20, 5, 5)
    (xsize, ysize, zsize) = (2., 3., 7.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), EmptyFF)

    # initialize some random scalar field
    source = -1. + 2. * np.random.rand(N1, N2, N3)

    # ensure the total is zero, otherwise Poisson's equation has no solution on the torus
    source -= np.sum(source) / (N1 * N2 * N3)

    # solve Poisson's equation
    soln = g.poisson_solve(source)

    # check that Poisson's equation is satisfied
    assert np.max(np.abs(source - g.lapl(soln))) < 1e-10

    # check the additional condition that the sum of soln is zero (which we artificially impose)
    assert abs(np.sum(soln)) < 1e-10


# test if wrapadound_array_dot works correctly on two simple one-dimensional arrays
# @test: wraparound_array_dot
def unit_test_5():
    print('-- unit test 5 --')

    A = np.array(range(10))  # [0, ..., 9]
    B = np.array([1, 1])
    assert wraparound_array_dot(A, B, [0]) == 1     # compute A[0] + A[1]
    assert wraparound_array_dot(A, B, [100]) == 1   # index 100 is the same as index 0 because 100 is divisible by 10
    assert wraparound_array_dot(A, B, [3]) == 7     # compute A[3] + A[4]
    assert wraparound_array_dot(A, B, [-27]) == 7   # index -27 is the same as index 3 because -27 = 3 + (-3) * 10
    assert wraparound_array_dot(A, B, [-1]) == 9    # compute A[-1] + A[0]
    assert wraparound_array_dot(A, B, [9]) == 9     # compute A[9] + A[0]

    B = np.array([1, 2, -3, -4, -5])
    assert wraparound_array_dot(A, B, [8]) == 12    # compute A[8] + 2 A [9] - 3 A[0] - 4 A[1] - 5 A[2]


# test if wrapadound_array_dot works correctly on two two-dimensional arrays
# @test: wraparound_array_dot
def unit_test_6():
    print('-- unit test 6 --')

    # initialize arrays
    A = np.zeros([7, 7])
    for i in range(7):
        for j in range(7):
            A[i, j] = i + 2 * j + 3 * i * j
    B = np.array([[1, 2],
                  [3, 4]])

    # check that when wraparound is not needed, the sum matches the analytical result
    for i in range(6):
        for j in range(6):
            assert wraparound_array_dot(A, B, [i, j]) == 28*i + 41*j + 30*i*j + 31

    # check the cases when one of the indexes wraps around
    for j in range(6):  # case i = 6, j < 6
        assert wraparound_array_dot(A, B, [6, j]) == 66 + 74*j
    for i in range(6):  # case i < 6, j = 6
        assert wraparound_array_dot(A, B, [i, 6]) == 109 + 82*i

    # check the case when both of the indexes wrap around
    assert wraparound_array_dot(A, B, [6, 6]) == 174


# test if wraparound_array_add works correctly on two simple one-dimensional arrays;
# assume it works for more complicated arrays as well because it's implemented
# very similarly to wraparound_array_dot
# @test: wraparound_array_add
def unit_test_7():
    print('-- unit test 7 --')

    A = np.zeros(10)
    B = np.array([1, 2, 3])

    wraparound_array_add(A, B, [0])  # add B at index 0
    assert np.array_equal(A, np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0]))

    wraparound_array_add(A, B, [201])  # index 201 is the same as index 1
    assert np.array_equal(A, np.array([1, 3, 5, 3, 0, 0, 0, 0, 0, 0]))

    wraparound_array_add(A, B, [-1])  # add B at index -1
    assert np.array_equal(A, np.array([3, 6, 5, 3, 0, 0, 0, 0, 0, 1]))

    wraparound_array_add(A, B, [8])  # add B at index 8
    assert np.array_equal(A, np.array([6, 6, 5, 3, 0, 0, 0, 0, 1, 3]))


# test if charge deposition works correctly on a particular example of a stationary partice,
# testing in particular if charge deposition correctly wraps around
# @test: Grid.deposit_charge, QuadraticSplineFF
def unit_test_8():
    print('-- unit test 8 --')

    (N1, N2, N3) = (7, 7, 7)
    (xsize, ysize, zsize) = (1., 2., 3.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    (Dx, Dy, Dz) = (xsize / N1, ysize / N2, zsize / N3)
    DV = Dx * Dy * Dz

    q = 1.  # charge
    dt = 0.1  # time step; doesn't affect anything but have to initialize

    # particle at the corner of eight cells
    R = np.array([0, 0, 0])

    rho_expected = np.zeros((N1, N2, N3))
    rho_expected[0, 0, 0] = rho_expected[0, 0, -1] = 1/8
    rho_expected[0, -1, 0] = rho_expected[0, -1, -1] = 1/8
    rho_expected[-1, 0, 0] = rho_expected[-1, 0, -1] = 1/8
    rho_expected[-1, -1, 0] = rho_expected[-1, -1, -1] = 1/8
    rho_expected *= q / DV

    g.refresh_charge()
    g.deposit_charge(q, R, R, dt)

    assert np.max(np.abs(rho_expected - g.rho)) < 1e-10


# test if charge deposition works correctly on a particular example of a stationary particle
# @test: Grid.deposit_charge, QuadraticSplineFF
def unit_test_9():
    print('-- unit test 9 --')

    (N1, N2, N3) = (7, 7, 7)
    (xsize, ysize, zsize) = (1., 2., 3.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    (Dx, Dy, Dz) = (xsize / N1, ysize / N2, zsize / N3)
    DV = Dx * Dy * Dz

    q = 1.  # charge
    dt = 0.1  # time step; doesn't affect anything but have to initialize

    R = np.array([3.3 * Dx, 4.6 * Dy, 3.65 * Dz])

    rho_expected = np.zeros((N1, N2, N3))
    for (i, f1) in [(2, 0.245), (3, 0.71), (4, 0.045)]:
        for (j, f2) in [(3, 0.08), (4, 0.74), (5, 0.18)]:
            for (k, f3) in [(2, 0.06125), (3, 0.7275), (4, 0.21125)]:
                rho_expected[i, j, k] = f1 * f2 * f3
    rho_expected *= q / DV

    g.refresh_charge()
    g.deposit_charge(q, R, R, dt)

    assert np.max(np.abs(rho_expected - g.rho)) < 1e-10


# test if charge deposition works correctly on a particular example of a moving particle
# @test: Grid.deposit_charge
def unit_test_10():
    print('-- unit test 10 --')

    (N1, N2, N3) = (7, 7, 7)
    (xsize, ysize, zsize) = (1., 2., 3.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    (Dx, Dy, Dz) = (xsize / N1, ysize / N2, zsize / N3)
    DV = Dx * Dy * Dz

    q = 123.45  # charge
    dt = 0.7  # time step

    R0 = np.array([3.3 * Dx, 4.6 * Dy, 3.65 * Dz])  # initial position
    R1 = np.array([3.5 * Dx, 4.0 * Dy, 4.0 * Dz])   # final position (after time dt)

    # compute the expected charge density when the particle is at position R0
    rho_0 = np.zeros((N1, N2, N3))
    for (i, f1) in [(2, 0.245), (3, 0.71), (4, 0.045)]:
        for (j, f2) in [(3, 0.08), (4, 0.74), (5, 0.18)]:
            for (k, f3) in [(2, 0.06125), (3, 0.7275), (4, 0.21125)]:
                rho_0[i, j, k] = f1 * f2 * f3
    rho_0 *= q / DV

    # compute the expected charge density when the particle is at position R1
    rho_1 = np.zeros((N1, N2, N3))
    for (i, f1) in [(2, 0.125), (3, 0.75), (4, 0.125)]:
        for (j, f2) in [(3, 0.5), (4, 0.5)]:
            for (k, f3) in [(3, 0.5), (4, 0.5)]:
                rho_1[i, j, k] = f1 * f2 * f3
    rho_1 *= q / DV

    g.refresh_charge()
    g.deposit_charge(q, R0, R1, dt)

    assert np.max(np.abs(rho_1 - g.rho)) < 1e-10  # charge density must be deposited with the new position

    assert np.max(np.abs((rho_1 - rho_0) / dt + g.divp(g.J))) < 1e-10  # check charge conservation, d/dt rho + div J = 0

    # check that the current vanishes sufficiently far on the torus
    assert np.max(np.abs(g.J[:, 0, :, :])) < 1e-10  # the current must be zero far away
    assert np.max(np.abs(g.J[:, :, 0, :])) < 1e-10  # the current must be zero far away
    assert np.max(np.abs(g.J[:, :, :, 0])) < 1e-10  # the current must be zero far away


# test if the Vay particle pusher works correctly on a specific example
# @test: Particle.vay_push
def unit_test_11():
    print('-- unit test 11 --')

    # initialize particle
    q = 2.
    m = 16.
    R0 = np.array([8.3, -2.7, 0.5])
    v = np.array([0.1, 0.2, 0.3])

    # initialize fields and time step
    E = np.array([1., 0., 0.])
    B = np.array([0, -0.5, -0.5])
    dt = 0.8

    # push particle
    ptc = Particle(m, q, R0, v=v)
    ptc.vay_push(dt, E, B)

    # test that even though R0 and v were passed to the particle init function, they were not modified
    assert np.max(np.abs(R0 - np.array([8.3, -2.7, 0.5]))) < 1e-10
    assert np.max(np.abs(v - np.array([0.1, 0.2, 0.3]))) < 1e-10

    # check if the result matches what is expected
    # (see Mathematica notebook for the calculation)
    R_expected = np.array([8.455496301546557, -2.53677332116603, 0.7313748403543063])
    v_expected = np.array([0.194370376933196, 0.2040333485424628, 0.2892185504428829])
    assert np.max(np.abs(ptc.R - R_expected)) < 1e-10
    assert np.max(np.abs(ptc.v - v_expected)) < 1e-10


# test if the Vay particle pusher works correctly on an example of a very fast particle and strong E, B
# @test: Particle.vay_push
def unit_test_12():
    print('-- unit test 12 --')

    # initialize particle
    q = 567.
    m = 234.
    R0 = np.array([190, 2.87, 0.354])
    v = np.array([-0.9, -0.25, 0.34])

    # initialize fields and time step
    E = np.array([123, 456, 789])
    B = np.array([900, 5284, -345])
    dt = 0.35

    # push particle
    ptc = Particle(m, q, R0, v=v)
    ptc.vay_push(dt, E, B)

    # test that even though R0 and v were passed to the particle init function, they were not modified
    assert np.max(np.abs(R0 - np.array([190, 2.87, 0.354]))) < 1e-10
    assert np.max(np.abs(v - np.array([-0.9, -0.25, 0.34]))) < 1e-10

    # check if the result matches what is expected
    # (see Mathematica notebook for the calculation)
    R_expected = np.array([190.20521054643586, 3.1078960452886824, 0.19974866329624416])
    v_expected = np.array([0.5863158469596151, 0.6797029865390921, -0.4407181048678738])
    assert np.max(np.abs(ptc.R - R_expected)) < 1e-10
    assert np.max(np.abs(ptc.v - v_expected)) < 1e-10


# test if the velocity remains unchanged if E + v x B = 0
# @test: Particle.vay_push
def unit_test_13():
    print('-- unit test 13 --')

    # initialize particle
    q = 567.
    m = 234.
    R0 = np.array([190, 2.87, 0.354])
    v = np.array([-0.9, -0.25, 0.34])

    # initialize fields and time step
    B = np.array([900, 5284, -345])
    dt = 0.35
    E = -np.cross(v, B)  # ensure that E + v x B = 0

    # push particle
    ptc = Particle(m, q, R0, v=v)
    ptc.vay_push(dt, E, B)

    # check if the result matches what is expected
    # (see Mathematica notebook for the calculation)
    R_expected = R0 + dt * v
    v_expected = v
    assert np.max(np.abs(ptc.R - R_expected)) < 1e-10
    assert np.max(np.abs(ptc.v - v_expected)) < 1e-10


# test that the gyroradius for precession in a constant magnetic field is correctly predicted
# @test: Particle.vay_push
def unit_test_14():
    print('-- unit test 14 --')

    # initialize particle
    q = 567.
    m = 234.
    R0 = np.zeros(3)
    v = np.array([0.5, 0, 0])

    # initialize simulation parameters
    Bz = 0.123
    gamma = 1 / np.sqrt(1 - np.dot(v, v))
    omega_c = q * Bz / (gamma * m)  # cyclotron frequency
    dt = (1/omega_c) * 2 / (1 + np.sqrt(2))  # pick dt such that the particle makes an octagon every 8 time steps

    # initialize fields
    E = np.zeros(3)
    B = np.array([0, 0, Bz])

    # initialize the particle and push it by four time steps
    ptc = Particle(m, q, R0, v=v)
    ptc.vay_push(dt, E, B)
    ptc.vay_push(dt, E, B)
    ptc.vay_push(dt, E, B)
    ptc.vay_push(dt, E, B)

    # check if the result matches what is expected
    # (see Mathematica notebook for the calculation)
    R__norm_expected = 2 * np.linalg.norm(v) / omega_c * np.sqrt(1 + (omega_c * dt / 2)**2)
    v_expected = -v
    assert abs(np.linalg.norm(ptc.R) - R__norm_expected) < 1e-10
    assert np.max(np.abs(ptc.v - v_expected)) < 1e-10

    # check that after eight time steps, the particle returns to the initial configuration
    ptc.vay_push(dt, E, B)
    ptc.vay_push(dt, E, B)
    ptc.vay_push(dt, E, B)
    ptc.vay_push(dt, E, B)
    assert abs(np.linalg.norm(ptc.R)) < 1e-10
    assert np.max(np.abs(ptc.v - v)) < 1e-10


# test if Newton's first law is satisfied for stationary particles,
# namely that a stationary charged particle does not experience a force from its own electrostatic field
# @test: Grid.get_EB_at, Grid.set_semistatic_init_conds, Grid.deposit_charge
def unit_test_15():
    print('-- unit test 15 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize particle parameters
    q = 1.
    R = np.array([0.21, 0.32, -3.897])

    # initialize time step; it doesn't affect anything here but have to initialize anyway
    dt = 0.1

    # deposit charge
    g.refresh_charge()
    g.deposit_charge(q, R, R, dt)

    # HACKY: to ensure in a simple way that total charge is zero, just introduce an ambient charge density
    g.rho -= np.sum(g.rho) / (N1 * N2 * N3)

    # set the initial conditions
    g.set_semistatic_init_conds()
    g.compute_EB_centered()

    # interpolate the fields at the position of the particle; the result should be zero
    (E, B) = g.get_EB_at(R)
    assert np.max(np.abs(E)) < 1e-10
    assert np.max(np.abs(B)) < 1e-10


# test that for a constant E and B, interpolation returns these constant values
# @test: Grid.get_EB_at
def unit_test_16():
    print('-- unit test 16 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # HACKY: manually set E and B to some constant values,
    # although Grid.semistatic_init_conds would not produce such conditions
    my_E = np.array([123, 456, 789])
    my_B = np.array([-9, 3, 20])
    g.E[:, :, :, :] = my_E[:, np.newaxis, np.newaxis, np.newaxis]
    g.B[:, :, :, :] = my_B[:, np.newaxis, np.newaxis, np.newaxis]
    g.compute_EB_centered()

    # pick some position and interpolate
    R = np.array([1.5, 9.3, 8.127])
    (E, B) = g.get_EB_at(R)

    # check that the values match
    assert np.max(np.abs(E - my_E)) < 1e-10
    assert np.max(np.abs(B - my_B)) < 1e-10


# test that Simulation.initialize_charges works as expected on a particular example
# @test: Simulation.__init__, Simulation.initialize_charges
def unit_test_17():
    print('-- unit test 17 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize some particles with zero total charge
    particles = [random_particle((xsize, ysize, zsize)) for i in range(10)]
    particles[-1].q = -sum(map(lambda ptc: ptc.q, particles[:-1]))

    # copy the particles list for later
    particles_copy = list(map(lambda ptc: ptc.copy(), particles))

    # choose a time step for initialization
    dt = 0.05

    # compute the expected charge and current densities
    g1 = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)
    g2 = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)
    g1.refresh_charge()
    g2.refresh_charge()
    for ptc in particles:
        R1 = ptc.R
        R0 = R1 - dt * ptc.v
        g1.deposit_charge(ptc.q, R0, R1, dt)
        g2.deposit_charge(ptc.q, R0, R0, dt)
    rho_1 = g1.rho  # charge density at time 0
    J_0 = g1.J      # current density at time -dt
    rho_0 = g2.rho  # charge density at time -dt

    # initialize charges and currents in the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges(dt)

    # check that the charge and current densities match what is expected, and that charge conservation holds
    assert np.max(np.abs(sim.grid.rho - rho_1)) < 1e-10
    assert np.max(np.abs(sim.grid.J - J_0)) < 1e-10
    assert np.max(np.abs((rho_1 - rho_0) / dt + sim.grid.divp(J_0))) < 1e-10  # d/dt rho + div J = 0

    # check that particles are not altered by Simulation initialization
    for (ptc, ptc1) in zip(particles, particles_copy):
        # check that all parameters are equal
        assert ptc.m == ptc1.m
        assert ptc.q == ptc1.q
        assert np.array_equal(ptc.R, ptc1.R)
        assert np.array_equal(ptc.v, ptc1.v)
        assert np.array_equal(ptc.u, ptc1.u)


# test Particle.copy on a particular example
# @test: Particle.copy
def unit_test_18():
    print('-- unit test 18 --')

    # initialize a particle with some parameters
    m = 123.
    q = 456.
    R = np.array([1.2, 6.7, -9.4])
    v = np.array([0.2, -0.4, 0.0])
    ptc = Particle(m, q, R, v=v)

    # make a copy
    ptc1 = ptc.copy()

    # check that the copy has the right parameters
    assert ptc1.m == m
    assert ptc1.q == q
    assert np.array_equal(ptc1.R, R)
    assert np.array_equal(ptc1.v, v)
    assert np.array_equal(ptc1.u, v / np.sqrt(1 - np.dot(v, v)))

    # modify the copy
    ptc1.m = 999
    ptc1.q = 999
    ptc1.R = np.random.rand(3)
    ptc1.u = np.random.rand(3)
    ptc1.compute_v()

    # check that the original particle still has the old parameters
    assert ptc.m == m
    assert ptc.q == q
    assert np.array_equal(ptc.R, R)
    assert np.array_equal(ptc.v, v)
    assert np.array_equal(ptc.u, v / np.sqrt(1 - np.dot(v, v)))


# test that Simulation.set_semistatic_init_conds maintains the divergence part of Maxwell's equations
# @test: Simulation.set_semistatic_init_conds
def unit_test_19():
    print('-- unit test 19 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize some particles with zero total charge
    particles = [random_particle((xsize, ysize, zsize)) for i in range(10)]
    particles[-1].q = -sum(map(lambda ptc: ptc.q, particles[:-1]))

    # choose a time step for initialization
    dt = 0.05

    # initialize charges and currents in the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges(dt)

    # initialize the fields E, B
    sim.set_semistatic_init_conds()

    # check that div E = rho and div B = 0
    assert np.max(np.abs(g.divp(g.E) - g.rho)) < 1e-10
    assert np.max(np.abs(g.divm(g.B))) < 1e-10


# test that initial conditions satisfy div E = rho and div B = 0
# @test: Grid.set_semistatic_init_conds, Grid.divp, Grid.divm
def unit_test_20():
    print('-- unit test 20 --')

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
    assert np.max(np.abs(g.divp(g.E) - g.rho)) < 1e-10  # div E - rho = 0 should hold to machine precision
    assert np.max(np.abs(g.divm(g.B))) < 1e-10          # div B = 0 should hold to machine precision


# test that Grid.make_divp_zero and Grid.make_divm_zero works correctly
# @test: make_divp_zero, make_divm_zero
def unit_test_21():
    print('-- unit test 21 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize a random vector field
    v = np.random.rand(3, N1, N2, N3)

    # derive from v similar fields which have divp and divm zero, respectively
    v1 = g.make_divp_zero(v)
    v2 = g.make_divm_zero(v)

    # check that divp v1 = 0 and divm v2 = 0
    assert np.max(np.abs(g.divp(v1))) < 1e-10
    assert np.max(np.abs(g.divm(v2))) < 1e-10


# test that Grid.get_magnetic_force correctly computes the magnetic force acting on a particle
# @test: Grid.get_magnetic_force
def unit_test_22():
    print('-- unit test 22 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # pick the particle's initial and final position, as well as the time step
    R0 = np.array([0.1, 0.2, 0.3])     # position at time t0
    R1 = np.array([0.11, 0.22, 0.35])  # position at time t1
    dt = 0.1

    # pick the particle's charge
    q = 1.

    # deposit charges, so that g.rho is now at time t1 and g.J is at time t0
    g.refresh_charge()
    g.deposit_charge(q, R0, R1, dt)

    # HACKY: make B some divergenceless vector field
    # this is to be interpreted as the field at time t0
    B0 = g.make_divm_zero(np.random.rand(3, N1, N2, N3))

    # compute the magnetic component of the Lorentz force acting on the particle in two different ways
    F1 = g.DV * np.sum(g.ccp(g.J, B0), axis=(1,2,3))
    F2 = g.get_magnetic_force(B0, q, R0, R1, dt)

    # assert that the two calculations give the same result
    assert np.linalg.norm(F1 - F2) < 1e-10


# test that Journal correctly records particle data
# @test: Journal.__init__, Journal.particle_push
def unit_test_23():
    print('-- unit test 23 --')

    # initialize a journal
    j = Journal()

    # pick the number of time steps and the number of particles
    nsteps = 100
    nparticles = 20

    my_particles_list = []
    for i in range(nsteps):
        # generate some random particles
        particles = [random_particle((1., 1., 1.)) for _ in range(nparticles)]

        # record in journal
        j.record_particles(particles)

        # record in my_particles_list
        my_particles_list.append([])
        for ptc in particles:
            my_particles_list[-1].append(Particle(ptc.m, ptc.q, ptc.R, v=ptc.v))

        # scramble the particles' data to make sure particles are copied into journal and not stored as pointers
        for ptc in particles:
            ptc.m += random.random()
            ptc.q += random.random()
            ptc.R += np.random.rand(3)
            ptc.v += np.random.rand(3)
            ptc.u += np.random.rand(3)

    # check that the journal data has the right number of entries
    assert len(j.particles_list) == nsteps

    # compare journal entries with what's expected
    for (particles1, particles2) in zip(j.particles_list, my_particles_list):
        # for each time step, iterate through the particles and compare
        for (ptc1, ptc2) in zip(particles1, particles2):
            assert ptc1.m == ptc2.m
            assert ptc1.q == ptc2.q
            assert np.linalg.norm(ptc1.R - ptc2.R) < 1e-10
            assert np.linalg.norm(ptc1.v - ptc2.v) < 1e-10
            assert np.linalg.norm(ptc1.u - ptc2.u) < 1e-10


# test that EverythingJournal correctly records particle and field data
# @test: EverythingJournal.__init__, EverythingJournal.record_particles, EverythingJournal.record_fields
def unit_test_24():
    print('-- unit test 24 --')

    # initialize a journal
    j = EverythingJournal()

    # pick the number of time steps, the number of particles, and the dimensions of field arrays
    nsteps = 100
    nparticles = 20
    (N1, N2, N3) = (10, 20, 30)

    my_particles_list = []
    my_E_list = []
    my_B_list = []
    for i in range(nsteps):
        # generate some random particles
        particles = [random_particle((1., 1., 1.)) for _ in range(nparticles)]

        # generate random E and B
        E = np.random.rand(3, N1, N2, N3)
        B = np.random.rand(3, N1, N2, N3)

        # record in journal
        j.record_particles(particles)
        j.record_fields(E, B)

        # record in my_particles_list
        my_particles_list.append([])
        for ptc in particles:
            my_particles_list[-1].append(Particle(ptc.m, ptc.q, ptc.R, v=ptc.v))

        # record in my_E_list, my_B_list
        my_E_list.append(E.copy())
        my_B_list.append(B.copy())

        # scramble the particles' data to make sure particles are copied into journal and not stored as pointers
        for ptc in particles:
            ptc.m += random.random()
            ptc.q += random.random()
            ptc.R += np.random.rand(3)
            ptc.v += np.random.rand(3)
            ptc.u += np.random.rand(3)

        # scramble the fields data to make sure particles are copied into journal and not stored as pointers
        E += np.random.rand(3, N1, N2, N3)
        B += np.random.rand(3, N1, N2, N3)

    # check that journal data has the right number of entries
    assert len(j.particles_list) == nsteps
    assert len(j.E_list) == nsteps
    assert len(j.B_list) == nsteps

    # compare journal particle entries with what's expected
    for (particles1, particles2) in zip(j.particles_list, my_particles_list):
        # for each time step, iterate through the particles and compare
        for (ptc1, ptc2) in zip(particles1, particles2):
            assert ptc1.m == ptc2.m
            assert ptc1.q == ptc2.q
            assert np.linalg.norm(ptc1.R - ptc2.R) < 1e-10
            assert np.linalg.norm(ptc1.v - ptc2.v) < 1e-10
            assert np.linalg.norm(ptc1.u - ptc2.u) < 1e-10

    # compare journal field entries with what's expected
    for (E1, E2) in zip(j.E_list, my_E_list):
        assert np.array_equal(E1, E2)
    for (B1, B2) in zip(j.B_list, my_B_list):
        assert np.array_equal(B1, B2)


# test that Simulation.run with EverythingJournal woks correctly
# @test: Simulation.run, EverythingJournal
def unit_test_25():
    print('-- unit test 25 --')

    # initialize a journal
    j = EverythingJournal()

    # initialize simulation parameters
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    nparticles = 20
    nsteps = 40

    # initialize the simulation
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)
    dt = 0.2 * g.max_time_step
    particles = [random_particle((xsize, ysize, zsize)) for _ in range(nparticles)]
    particles[-1].q = -sum([ptc.q for ptc in particles[:-1]])  # make sure the total charge is zero
    sim = Simulation(g, particles)
    sim.initialize_charges(dt)
    sim.set_semistatic_init_conds()

    # initialize a copy for later
    g_copy = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)
    particles_copy = [ptc.copy() for ptc in particles]
    sim_copy = Simulation(g_copy, particles_copy)
    sim_copy.initialize_charges(dt)
    sim_copy.set_semistatic_init_conds()

    # run sim for nsteps, recording the data in my_particles_list, my_E_list, my_B_list
    my_particles_list = [[ptc.copy() for ptc in particles]]
    my_E_list = [g.E.copy()]
    my_B_list = [g.B.copy()]
    for i in range(nsteps):
        sim.vay_make_step(dt)

        my_particles_list.append([ptc.copy() for ptc in particles])
        my_E_list.append(g.E.copy())
        my_B_list.append(g.B.copy())

    # reproduce the same with sim_copy.run
    sim_copy.run(nsteps, dt, j, pusher="vay")

    # assert that the hournal has the right number of entries
    assert len(j.particles_list) == len(j.E_list) == len(j.B_list) == nsteps + 1  # initial step plus nsteps

    # compare journal particle entries with what's expected
    for (particles1, particles2) in zip(j.particles_list, my_particles_list):
        # for each time step, iterate through the particles and compare
        for (ptc1, ptc2) in zip(particles1, particles2):
            assert ptc1.m == ptc2.m
            assert ptc1.q == ptc2.q
            assert np.linalg.norm(ptc1.R - ptc2.R) < 1e-10
            assert np.linalg.norm(ptc1.v - ptc2.v) < 1e-10
            assert np.linalg.norm(ptc1.u - ptc2.u) < 1e-10

    # compare journal field entries with what's expected
    for (E1, E2) in zip(j.E_list, my_E_list):
        assert np.array_equal(E1, E2)
    for (B1, B2) in zip(j.B_list, my_B_list):
        assert np.array_equal(B1, B2)


# test that Simulation.vay_make_step maintains charge conservation and Maxwell's equations,
# thus testing the field part of the entire PIC loop with the Vay pusher
def intg_test_0():
    print('-- integration test 0 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize some particles with zero total charge
    particles = [random_particle((xsize, ysize, zsize)) for i in range(10)]
    particles[-1].q = -sum(map(lambda ptc: ptc.q, particles[:-1]))

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges()
    sim.set_semistatic_init_conds()

    # run the simulation for 10 time steps
    for i in range(10):
        # @pre: g.rho, g.E, g.B, ptc.R, ptc.v are at time 0,
        # and g.J is at time -1
        rho_old = g.rho.copy()
        E_old = g.E.copy()
        B_old = g.B.copy()

        # pick a random time step which is not too large
        # NOTE: the time step is allowed to vary in simulations, so we randomize it
        dt = (0.01 + 0.98 * random.random()) * g.max_time_step

        # make one time step, so that
        # now g.rho, g.E, g.B, ptc.R, ptc.v are at time 1,
        # and g.J is at time 0
        sim.vay_make_step(dt)

        # check that Maxwell's equations are satisfied
        assert np.max(np.abs(g.divm(g.B))) < 1e-10                                # div^- B = 0
        assert np.max(np.abs(g.divp(g.E) - g.rho)) < 1e-10                        # div^+ E = rho
        assert np.max(np.abs((g.B - B_old) / dt + g.curlm(g.E))) < 1e-10          # d_t^- B + curl^- E = 0
        assert np.max(np.abs((g.E - E_old) / dt - g.curlp(B_old) + g.J)) < 1e-10  # d_t^+ E - curl^+ B = -J

        # check charge conservation
        assert np.max(np.abs((g.rho - rho_old) / dt + g.divp(g.J))) < 1e-10  # (rho_1 - rho_0) / dt + div^+ J_0 = 0


# test that when two particles on opposite sides of the torus are initially stationary, they remain so for all time
def intg_test_1():
    print('-- integration test 1 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize two stationary particles of opposite charge on opposite sides of the torus
    Ra = np.zeros(3)
    Rb = np.array([xsize/2, ysize/2, zsize/2])
    particles = [
        Particle(1.,  1., Ra),
        Particle(1., -1., Rb)
    ]

    # pick a time step
    dt = 0.05

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges()
    sim.set_semistatic_init_conds()

    # run for 10 time steps with the Vay pusher
    for i in range(10):
        sim.vay_make_step(dt)

    # check that the particles haven't moved
    assert np.max(np.abs(particles[0].R - Ra)) < 1e-10
    assert np.max(np.abs(particles[1].R - Rb)) < 1e-10


# test that field momentum is conserved when there are no charges
def intg_test_2():
    print('-- integration test 2 --')

    # initialize a grid
    (N1, N2, N3) = (10, 20, 7)
    (xsize, ysize, zsize) = (1., 3., 8.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize zero particles
    particles = []

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges()
    sim.set_semistatic_init_conds()

    # HACKY: initialize E and B to some divergenceless random fields
    g.E = g.make_divp_zero(np.random.rand(3, N1, N2, N3))
    g.B = g.make_divm_zero(np.random.rand(3, N1, N2, N3))

    # pick a time step
    dt = 0.5 * g.max_time_step

    # initialize a list to store the momentum values
    plist = []

    # run the simulation for 10 time steps
    for i in range(10):
        B_old = g.B.copy()

        # two equivalent options are sim.make_step(dt) and g.evolve_fields(dt)
        sim.vay_make_step(dt)

        # if enter plist.append(g.DV * np.sum(g.cross(g.E, B_old), axis=(1, 2, 3))) instead, the test fails
        plist.append(g.DV * np.sum(g.ccp(g.E, B_old), axis=(1,2,3)))

    # check that all the elements of plist are equal
    assert max([max([np.linalg.norm(x - y) for y in plist]) for x in plist]) < 1e-10


# test that the time difference of the field momentum is minus the total Lorentz force, as computed by a certain formula
def intg_test_3():
    print('-- integration test 3 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize some particles with zero total charge
    particles = [random_particle((xsize, ysize, zsize)) for i in range(10)]
    particles[-1].q = -sum(map(lambda ptc: ptc.q, particles[:-1]))

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges()
    sim.set_semistatic_init_conds()

    # initialize the field momentum to a null value
    p = np.zeros(3)

    # pick a time step
    dt = (0.01 + 0.98 * random.random()) * g.max_time_step

    # run the simulation for 10 time steps
    for i in range(10):
        # @pre: g.rho, g.E, g.B, ptc.R, ptc.v are at time 0,
        # and g.J is at time -1
        rho_old = g.rho.copy()
        E_old = g.E.copy()
        B_old = g.B.copy()

        # store the old field momentum
        p_old = p.copy()

        # evolve the field by some time step
        sim.vay_make_step(dt)

        # compute the current field momentum
        p = g.DV * np.sum(g.ccp(g.E, B_old), axis=(1,2,3))

        # after the initial time step, compare p to p_old
        if i > 0:
            # compute the rate of change of the field momentum
            F = (p - p_old) / dt

            # compute what the rate of change is expected to be
            Ec = E_old.copy()
            Ec[0] = 0.5 * (Ec[0] + np.roll(Ec[0], -1, axis=0))
            Ec[1] = 0.5 * (Ec[1] + np.roll(Ec[1], -1, axis=1))
            Ec[2] = 0.5 * (Ec[2] + np.roll(Ec[2], -1, axis=2))
            F_expected = -g.DV * np.sum(rho_old * Ec + g.ccp(g.J, B_old), axis=(1,2,3))

            # check that dp/dt matches the expectation
            assert np.linalg.norm(F - F_expected) < 1e-10


# test that the time difference of the Poynting vector is minus the Lorentz force density plus a field gradient term
def intg_test_4():
    print('-- integration test 4 --')

    # initialize a grid
    (N1, N2, N3) = (10, 20, 7)
    (xsize, ysize, zsize) = (1., 3., 8.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize some particles with zero total charge
    particles = [random_particle((xsize, ysize, zsize)) for i in range(10)]
    particles[-1].q = -sum(map(lambda ptc: ptc.q, particles[:-1]))

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges()
    sim.set_semistatic_init_conds()

    # initialize some particles with zero total charge
    particles = [random_particle((xsize, ysize, zsize)) for i in range(10)]
    particles[-1].q = -sum(map(lambda ptc: ptc.q, particles[:-1]))

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges()
    sim.set_semistatic_init_conds()

    # pick a time step
    dt = (0.01 + 0.98 * random.random()) * g.max_time_step

    # initialize the Poynting vector field to a null value
    S = np.zeros((3, N1, N2, N3))

    # run the simulation for 10 time steps
    for iteration in range(10):
        # record some quantities from the previous cycle
        rho_old = g.rho.copy()  # charge density
        E_old = g.E.copy()      # electric field
        Ec_old = g.Ec.copy()    # electric field, centered
        B_old = g.B.copy()      # magnetic field
        S_old = S               # Poynting vector

        # run the simulation for one time step
        sim.vay_make_step(dt)

        # compute the Poynting vector everywhere on the grid
        S = g.ccp(g.E, B_old)

        # after the initial time step, compare S to S_old
        if iteration > 0:
            dSdt = (S - S_old) / dt

            # compute a certain term which is the gradient of a certain field quantity (see the writeup)
            field_term = np.zeros((3, N1, N2, N3))
            for i in range(3):
                for j in range(3):
                    field_term[i] -= 0.5 * g.pdm(i,
                        E_old[j] * g.shiftp(i, E_old[j]) + g.shiftp(i, B_old[j])**2
                    )
                    field_term[i] += g.pdm(j,
                        g.shiftp(j, E_old[j]) * g.half_shiftp(i, E_old[i]) +
                        g.shiftp(j, B_old[i]) * g.half_shiftp(i, B_old[j])
                    )

            # compute the Lorentz force density
            lorentz = rho_old * Ec_old + g.ccp(g.J, B_old)

            # check that the time difference of the Poynting vector is what we expect
            assert np.max(np.abs(dSdt - field_term + lorentz)) < 1e-10


# check that the field induced by charge (q, R) at -R is the same as the field induced by (-q, -R) at R
# (implicitly assuming that the field induced by a charge at its own location is zero)
def intg_test_5():
    print('-- integration test 5 --')

    # initialize a grid
    (N1, N2, N3) = (20, 30, 40)
    (xsize, ysize, zsize) = (1.5, 2.6, 3.7)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # pick a charge and displacement vector
    q = 1.
    R = np.array([0.4, 0.8, -0.3])

    # pick a time step; the exact value doesn't matter but have to pick
    dt = 0.01

    # deposit charge from q at R and -q at -R
    g.refresh_charge()
    g.deposit_charge(q, R, R, dt)
    g.deposit_charge(-q, -R, -R, dt)

    # compute E and B
    g.set_semistatic_init_conds()
    g.compute_EB_centered()

    # extract E at R and -R
    E1 = g.get_EB_at(R)[0]
    E2 = g.get_EB_at(-R)[0]

    # assert that the field induced by (-q, -R) at R is the same as the field induced by (q, R) at -R
    assert np.linalg.norm(E1 - E2) < 1e-10


# repeat intg_test_0 with the xc pusher
def intg_test_6():
    print('-- integration test 6 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize some particles with zero total charge
    particles = [random_particle((xsize, ysize, zsize)) for i in range(10)]
    particles[-1].q = -sum(map(lambda ptc: ptc.q, particles[:-1]))

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges()
    sim.set_semistatic_init_conds()

    # run the simulation for 10 time steps
    for i in range(10):
        # @pre: g.rho, g.E, g.B, ptc.R, ptc.v are at time 0,
        # and g.J is at time -1
        rho_old = g.rho.copy()
        E_old = g.E.copy()
        B_old = g.B.copy()

        # pick a random time step which is not too large
        # NOTE: the time step is allowed to vary in simulations, so we randomize it
        dt = (0.01 + 0.98 * random.random()) * g.max_time_step

        # make one time step, so that
        # now g.rho, g.E, g.B, ptc.R, ptc.v are at time 1,
        # and g.J is at time 0
        sim.xc_make_step(dt)

        # check that Maxwell's equations are satisfied
        assert np.max(np.abs(g.divm(g.B))) < 1e-10                                # div^- B = 0
        assert np.max(np.abs(g.divp(g.E) - g.rho)) < 1e-10                        # div^+ E = rho
        assert np.max(np.abs((g.B - B_old) / dt + g.curlm(g.E))) < 1e-10          # d_t^- B + curl^- E = 0
        assert np.max(np.abs((g.E - E_old) / dt - g.curlp(B_old) + g.J)) < 1e-10  # d_t^+ E - curl^+ B = -J

        # check charge conservation
        assert np.max(np.abs((g.rho - rho_old) / dt + g.divp(g.J))) < 1e-10  # (rho_1 - rho_0) / dt + div^+ J_0 = 0


# test that the xc pusher preserves total momentum
def intg_test_7():
    print('-- integration test 7 --')

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize some particles with zero total charge
    particles = [random_particle((xsize, ysize, zsize)) for i in range(10)]
    particles[-1].q = -sum(map(lambda ptc: ptc.q, particles[:-1]))

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges()
    sim.set_semistatic_init_conds()

    # initialize the field momentum to a null value
    p = np.zeros(3)

    # pick a time step
    dt = (0.01 + 0.98 * random.random()) * g.max_time_step

    # run the simulation for 10 time steps
    for i in range(10):
        B_old = g.B.copy()

        # store the old field momentum
        p_old = p.copy()

        # evolve the field by one time step
        sim.xc_make_step(dt)

        # compute the total field and particles' momentum
        p = g.DV * np.sum(g.ccp(g.E, B_old), axis=(1,2,3)) + sum([ptc.get_total_momentum() for ptc in particles])

        # after the initial time step, compare p to p_old
        if i > 0:
            # check that dp/dt is zero
            F = (p - p_old) / dt
            assert np.linalg.norm(F) < 1e-10


# test if the field of two static charges is close to what we expect in the continuous case
def soft_test_0():
    # initialize a grid
    g = Grid((200, 200, 200), (1., 1., 1.), QuadraticSplineFF)

    # initialize two particles of opposite charge
    q = 1.
    x = 0.1
    particles = [
        Particle(1., q, np.array([-x, 0, 0])),
        Particle(1., -q, np.array([x, 0, 0]))
    ]  # a positive particle on the negative x axis and a negative particle on the positive x axis

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges()
    sim.set_semistatic_init_conds()

    # compare fields to expectation
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
    # initialize a grid
    g = Grid((200, 200, 200), (1., 1., 1.), QuadraticSplineFF)

    # initialize the parameters of a ring of total charge Q, with its center at (0, 0, -h) and having radius R,
    # rotating with angular velocity omega;
    # also there is an identical ring but with charge -Q and not rotating
    Q = 1.
    R = 0.2
    h = 0.12
    omega = 4.5

    # initialize the number of particles
    N = 40
    q = Q / N

    # create the ring from particles
    particles = [
        Particle(1., q, np.array([R * np.cos(th), R * np.sin(th), -h]),
                        np.array([-R * omega * np.sin(th), R * omega * np.cos(th), 0]))
    for th in 2 * np.pi / N * np.arange(0, N)]
    particles += [
        Particle(1., -q, np.array([R * np.cos(th), R * np.sin(th), -h]),
                        np.zeros(3))
    for th in 2 * np.pi / N * np.arange(0, N)]

    # pick a time step
    dt = 0.01 * g.max_time_step

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges(dt)
    sim.set_semistatic_init_conds()

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


# test if the electrostatic field is collinear with the displacement vector
def soft_test_2():
    print('-- soft test 2 --')

    # initialize a grid
    (N1, N2, N3) = (40, 40, 40)
    (xsize, ysize, zsize) = (1., 1., 1.)
    (Dx, Dy, Dz) = (xsize / N1, ysize / N2, zsize / N3)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # pick a time step; the value won't matter but have to pick
    dt = 0.05

    # pick a charge
    q = 1.

    # calculate the field that particle -q at position -R induces at position R
    def get_E_at_R(R):
        # deposit charge from two stationary particles of opposite charges located at R and -R
        g.refresh_charge()
        g.deposit_charge(q, R, R, dt)
        g.deposit_charge(-q, -R, -R, dt)

        # compute E and B (and B will be zero because the particles are not moving)
        g.set_semistatic_init_conds()
        g.compute_EB_centered()
        (E, B) = g.get_EB_at(R)
        return E

    # calculate the "degree of perpendicularity" of E to R (ideally zero)
    def f(a, b):
        R = np.array([Dx * a, Dy * b, 0])
        E = get_E_at_R(R)
        if np.linalg.norm(E) < 1e-10 or np.linalg.norm(R) < 1e-10:
            return 0.
        else:
            return np.linalg.norm(np.cross(E, R)) / np.linalg.norm(E) / np.linalg.norm(R)

    # Create a grid of points
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    f = np.vectorize(f)
    Z = f(X, Y)

    print('Plot of non-collinearity of electrostatic force')
    print('Numbers of grid points in each direction: (N1, N2, N3) =',  (N1, N2, N3))
    print('Median non-collinearity within the region:', np.median(Z))
    print('Mean:', np.mean(Z))

    # Plot the function
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(label='Value')
    plt.xlabel('x / Dx')
    plt.ylabel('y / Dy')
    plt.title('|E x R| / |E| / |R|')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure the aspect ratio is 1
    plt.show()


# two particles of opposite charge and equal mass orbiting each other
# @test: Grid.set_semistatic_init_conds, Grid.evolve_fields, Grid.deposit_charge, Particle.vay_push
def soft_test_3():
    print('-- soft test 3 --')

    # initialize a grid
    (N1, N2, N3) = (80, 80, 80)
    L = 1.
    (xsize, ysize, zsize) = (L, L, L)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # set up two particles which should orbit each other in a circle
    q = 0.5
    m = 1.
    R_mag = L/8
    v_mag = 0.75 * np.sqrt((q**2 / 4 / np.pi) / (4 * m * R_mag))  # the 0.75 factor is just put in empirically
    R = np.array([0, 1, 0]) * R_mag
    v = np.array([1, 0, 0]) * v_mag
    particles = [
        Particle(m, q, R, v=v),
        Particle(m, -q, -R, v=-v)
    ]

    # pick a time step which is not too large
    dt = min(0.1 * R_mag / v_mag, 0.5 * g.max_time_step)

    # initialize the simulation

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges(dt)
    sim.set_semistatic_init_conds()

    rlist = []
    xlist = []
    ylist = []

    # run the simulation for some number of time steps
    for i in range(1000):
        print(i)
        rlist.append(np.linalg.norm(particles[0].R - particles[1].R) / 2 / R_mag)
        xlist.append((particles[0].R - particles[1].R)[0])
        ylist.append((particles[0].R - particles[1].R)[1])

        sim.vay_make_step(dt)

    plt.plot(rlist)
    plt.plot(xlist)
    plt.plot(ylist)
    plt.show()



# test that Simulation.run with EverythingJournal woks correctly
# @test: Simulation.run, EverythingJournal
def soft_test_4():
    print('-- soft test 4 --')

    # initialize a journal
    j = ParticlesOnlyJournal()

    # initialize a grid
    (N1, N2, N3) = (10, 10, 10)
    (xsize, ysize, zsize) = (1., 1., 1.)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # pick a time step
    dt = 0.2 * g.max_time_step

    # pick the number of time steps
    nsteps = 100

    # initialize particles
    nparticles = 20
    particles = [random_particle((xsize, ysize, zsize)) for _ in range(nparticles)]
    particles[-1].q = -sum([ptc.q for ptc in particles[:-1]])  # make sure the total charge is zero

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges(dt)
    sim.set_semistatic_init_conds()

    # run
    sim.run(nsteps, dt, j, pusher="vay")

    # display
    j.to_gif((xsize, ysize, zsize), 0.3)
