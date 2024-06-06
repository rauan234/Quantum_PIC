import numpy as np
import random

from Grid import *
from Particle import *

# ------------------------------------------------------------------------------
#                                 PARTICLE
# ------------------------------------------------------------------------------


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

        self.E_B_initialized = True

    # do one interation of the PIC loop
    def make_step(self, dt):
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
            ptc.push(dt, E, B)
            qR0R1list.append((ptc.q, R0, ptc.R))
        # currently:
        # quantity  time
        # g.rho     0
        # g.J       -1
        # g.E       0
        # g.B       0
        # ptc.R     1
        # ptc.v     1

        # update charge and current densities, as well as field strengths
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

        self.grid.evolve_fields(dt)
        # currently:
        # quantity  time
        # g.rho     1
        # g.J       0
        # g.E       1
        # g.B       1
        # ptc.R     1
        # ptc.v     1
