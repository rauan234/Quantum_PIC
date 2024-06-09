import numpy as np
import random

from Grid import *
from Particle import *

# ------------------------------------------------------------------------------
#                                 SIMULATION
# ------------------------------------------------------------------------------
# The purpose of this class is to assemble the entire PIC loop in a compact and
# simple way that does not require the user to get into the weeds of the PIC
# loop, and thus prevents any mistakes that might occur. Each instance of the
# class stores a grid and a list of particles. During each run of make_step,
# both the grid and the particles are updated according to the PIC loop.
#
# RELIES ON:
#   [*] Grid
#   [*] Particle
#
# POSSIBLE ADDITIONS:
#   [*] another function for setting the initial conditions.
#   [*] change the particle pushing step to achieve better time centering.
# other than these two possible additions, this class seems pretty complete;
# I don't expect to add much to this class, as its main purpose (encapsulating
# the complexity of the PIC loop) has already been accomplished.


class Simulation:
    def __init__(self, grid, particles):
        assert isinstance(grid, Grid)
        assert isinstance(particles, list) and all(map(lambda x: isinstance(x, Particle), particles))
        self.grid = grid
        self.particles = particles

        self.rho_J_initialized = False
        self.E_B_initialized = False

    # deposit the initial charge and current densities onto the grid
    # here dt is the time step to which the initial conditions will be tuned; its precise value is not too consequential
    def initialize_charges(self, dt=None):
        if isinstance(dt, type(None)):
            dt = 0.5 * self.grid.max_time_step  # choose a default value for dt
        else:
            assert 0 < dt <= self.grid.max_time_step

        assert not self.rho_J_initialized  # if already initialized, should not need to initialize again
        self.rho_J_initialized = True

        self.grid.refresh_charge()  # set rho = J = 0
        for ptc in self.particles:
            R1 = ptc.R
            R0 = R1 - dt * ptc.v
            self.grid.deposit_charge(ptc.q, R0, R1, dt)  # move each particle and deposit charge and current

    # solve for E and B in a particular way
    def set_semistatic_init_conds(self):
        assert self.rho_J_initialized

        self.grid.set_semistatic_init_conds()
        self.grid.compute_EB_centered()

        self.E_B_initialized = True

    # do one interation of the PIC loop with the Vay pusher
    def vay_make_step(self, dt):
        assert self.rho_J_initialized and self.E_B_initialized

        # currently:
        # quantity  time
        # g.rho     0
        # g.J       -1
        # g.E       0
        # g.B       0
        # ptc.R     0
        # ptc.v     0

        # change the particles' momentum and position, recording their previous position in qR0R1list
        qR0R1list = []
        for ptc in self.particles:
            # interpolate E and B
            (E, B) = self.grid.get_EB_at(ptc.R)

            # IMPORTANT: first update velocity, then use this updated velocity to modify position
            # making ptc = (q, R_{t+1}, v_{t+1}),
            # where (IMPORTANT) R_{t+1} = R_t + dt * v_{t+1}, otherwise cannot guarantee charge conservation
            R0 = ptc.R.copy()
            ptc.vay_push(dt, E, B)
            qR0R1list.append((ptc.q, R0, ptc.R))
        # currently:
        # quantity  time
        # g.rho     0
        # g.J       -1
        # g.E       0
        # g.B       0
        # ptc.R     1
        # ptc.v     1

        # update the charge and current densities
        self.grid.refresh_charge()
        for (q, R0, R1) in qR0R1list:
            self.grid.deposit_charge(q, R0, R1, dt)
        # currently:
        # quantity  time
        # g.rho     1
        # g.J       0
        # g.E       0
        # g.B       0
        # ptc.R     1
        # ptc.v     1

        # evolve the electric and magnetic fields
        self.grid.evolve_fields(dt)
        self.grid.compute_EB_centered()
        # currently:
        # quantity  time
        # g.rho     1
        # g.J       0
        # g.E       1
        # g.B       1
        # ptc.R     1
        # ptc.v     1

    # do one interation of the PIC loop with the xc pusher, described in the writeup
    def xc_make_step(self, dt):
        assert self.rho_J_initialized and self.E_B_initialized

        # currently:
        # quantity   time
        # g.rho      0
        # g.J        -1
        # g.E        0
        # g.B        0
        # ptc.R      0
        # ptc.v      0
        # ptc.delta  0

        # change the particles' momentum and position, recording their previous position in qR0R1list
        qR0R1list = []
        for ptc in self.particles:
            # interpolate E and B
            (E, B) = self.grid.get_EB_at(ptc.R)

            # IMPORTANT: first update velocity, then use this updated velocity to modify position
            # making ptc = (q, R_{t+1}, v_{t+1}),
            # where (IMPORTANT) R_{t+1} = R_t + dt * v_{t+1}, otherwise cannot guarantee charge conservation
            R0 = ptc.R.copy()
            ptc.xc_push(dt, E)
            qR0R1list.append((ptc.q, R0, ptc.R))
        # currently:
        # quantity   time
        # g.rho      0
        # g.J        -1
        # g.E        0
        # g.B        0
        # ptc.R      1
        # ptc.v      1
        # ptc.delta  0

        # update the charge and current densities
        self.grid.refresh_charge()
        for (q, R0, R1) in qR0R1list:
            self.grid.deposit_charge(q, R0, R1, dt)
        # currently:
        # quantity   time
        # g.rho      1
        # g.J        0
        # g.E        0
        # g.B        0
        # ptc.R      1
        # ptc.v      1
        # ptc.delta  0

        # update the particles' momentum debt
        for (ptc, (q, R0, R1)) in zip(self.particles, qR0R1list):
            F = self.grid.get_magnetic_force(self.grid.B, q, R0, R1, dt)
            ptc.update_momentum_debt(dt, F)

        # evolve the electric and magnetic fields
        self.grid.evolve_fields(dt)
        self.grid.compute_EB_centered()
        # currently:
        # quantity   time
        # g.rho      1
        # g.J        0
        # g.E        1
        # g.B        1
        # ptc.R      1
        # ptc.v      1
        # ptc.delta  1
