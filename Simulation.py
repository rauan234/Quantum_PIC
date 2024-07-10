import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

import numpy as np
import random
import math

from Grid import *
from Particle import *


# ------------------------------------------------------------------------------
#                                   JOURNAL
# ------------------------------------------------------------------------------
# A class to record the states of the fields and particles in the simulation as
# it runs. Recording the fields at every grid point at every time step would be
# too expensive, but using Journal, we only record the field values we need.
#
# Class Journal itself is very small. It has the following subclasses:
#   [*] ParticlesOnlyJournal         - stores no field values
#   [*] EverythingJournal            - stores all field values at all grid points
#   [*] EnergyDensityInZPlaneJournal - stores energy density in a given Z plane


class Journal:
    def __init__(self):
        self.particles_list = []

    def record_particles(self, particles):
        # copy all particle data to self.particles_list
        # particle data includes:
        #   [*] mass
        #   [*] charge
        #   [*] position
        #   [*] velocity, momentum
        self.particles_list.append([ptc.copy() for ptc in particles])

    def record_fields(self, E, B):
        raise NotImplementedError

    def to_gif(self, *params):
        raise NotImplementedError


# store only particle data
class ParticlesOnlyJournal(Journal):
    def __init__(self):
        super().__init__()

    def record_fields(self, E, B):
        # do not record any field values
        pass

    # plot the motion of the particles projected onto plane z = z0
    # filename:         name of the file into which the gif will be saved
    # frame_rate:       frames per second of the gif
    # duration:         the total number of seconds of the gif
    # max_marker_size:  radius (in pixels) of the circles representing particles which are exactly in plane
    # display:          whether the gif will be displayed, in addition to being saved
    def to_gif(self, grid_size, z0, filename='particles_animation.gif', frame_rate=30, duration=5.,
               max_marker_size=10., display=True):
        # grid_size is the physical size of the grid, say in meters
        (xsize, ysize, zsize) = grid_size
        assert isinstance(xsize, float) and xsize > 0
        assert isinstance(ysize, float) and ysize > 0
        assert isinstance(zsize, float) and zsize > 0

        # z0 specifies the plane [0, xsize] x [0, ysize] x {z0} onto which the particles' motion is projected
        # note: there is no restriction on the range of z0 because the torus wraps around
        assert isinstance(z0, float)
    
        # create a figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(0, xsize)
        ax.set_ylim(0, ysize)
        ax.set_aspect(xsize / ysize)

        # ASSUMING: the number of particles does not change from one time step to another
        nparticles = len(self.particles_list[0])

        # ASSUMING: the ordering of the particles, and the particles' charges, do not change
        qlist = [ptc.q for ptc in self.particles_list[0]]

        # initialize dots to represent the particles
        dots = [ax.plot([], [], 'o', markersize=max_marker_size)[0] for _ in range(nparticles)]
    
        def init():
            for (dot, q) in zip(dots, qlist):
                dot.set_data([], [])
                dot.set_color(cm.coolwarm(0.5 + 0.5 * np.tanh(q)))  # color particles depending on their charge
            return dots
    
        def update(frame_ind):
            # given the frame index, calculate the original time series index
            ind = math.floor(nsteps * frame_ind / (nframes - 1))

            # get the list of particles' positions at the desired time
            Rlist = [ptc.R for ptc in self.particles_list[ind]]

            # compute the horizontal and vertical positions using the projection matrix,
            # modding out by xsize and ysize to take into account that this is on a torus
            screen_positions = np.array([R[:-1] for R in Rlist])
            screen_positions[:, 0] = screen_positions[:, 0] % xsize
            screen_positions[:, 1] = screen_positions[:, 1] % ysize

            # compute the distances of the points to the screen using the normal vector
            heights = np.array([R[-1] - z0 for R in Rlist])
            # rescale by grid size
            heights *= 2 * np.pi / zsize

            # update points on the screen
            for (pos, h, dot) in zip(screen_positions, heights, dots):
                # place the dot representing the particle at the appropriate position
                dot.set_data(pos)

                # update size based on distance to the screen, particles closer to the screen being biggest
                # and particles opposite to the plane having size sqrt(0.1) * max_marker_size
                msize = max_marker_size * np.sqrt(0.55 + 0.45 * np.cos(h))
                dot.set_markersize(msize)
            return dots

        # time interval between two consecutive frames, equal to 1000 milliseconds floor-divided by frame rate
        interval = 1000 // frame_rate

        # the total number of frames
        nframes = math.floor(duration * frame_rate)

        # the total number of time steps which were simulated; subtract 1 because not counting the initial step
        nsteps = len(self.particles_list) - 1
    
        # create the animation
        ani = animation.FuncAnimation(fig, update, frames=nframes, init_func=init, blit=True, interval=interval)

        # save the animation as a GIF
        ani.save(filename, writer='pillow')

        if display:
            # display the GIF
            plt.show()


# store particle data and also the field values at all grid points
class EverythingJournal(Journal):
    def __init__(self):
        super().__init__()
        self.E_list = []
        self.B_list = []

    def record_fields(self, E, B):
        # record all E and B data
        self.E_list.append(np.copy(E))
        self.B_list.append(np.copy(B))

    def to_gif(self):
        pass


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

    # do one interation of the PIC loop with the Vay pusher, using quantum circuits
    def q_vay_make_step(self, dt):
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
        self.grid.q_evolve_fields(dt)
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

    # run the simulation for nsteps and record it in journal
    def run(self, nsteps, dt, journal, pusher="vay"):
        assert isinstance(nsteps, int) and nsteps > 0
        assert isinstance(dt, float) and 0 < dt <= self.grid.max_time_step
        assert isinstance(journal, Journal)

        # record the initial field values and particle states
        journal.record_fields(self.grid.E, self.grid.B)
        journal.record_particles(self.particles)

        # run for nsteps and record in journal
        if pusher == "vay":
            # use Vay particle pusher
            for i in range(nsteps):
                self.vay_make_step(dt)
                journal.record_fields(self.grid.E, self.grid.B)
                journal.record_particles(self.particles)
        else:
            raise ValueError("Unrecognized pusher")
