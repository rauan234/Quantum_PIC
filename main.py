import matplotlib.pyplot as plt
import numpy as np
import random

import matplotlib.animation as animation
from matplotlib import cm
import imageio

from Grid import *
from Particle import *
from FormFactor import *
from tests import *


def dot_moving_in_a_circle_example():
    # CREDIT: ChatGPT 4o

    # Define the number of frames for the animation
    num_frames = 200

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # Create a red dot
    dot, = ax.plot([], [], 'ro')

    # Function to initialize the animation
    def init():
        dot.set_data([], [])
        return dot,

    # Function to update the animation
    def update(frame):
        angle = 2 * np.pi * frame / num_frames
        x = np.cos(angle)
        y = np.sin(angle)
        dot.set_data(x, y)
        return dot,

    frame_rate = 60  # Frames per second
    interval = 1000 // frame_rate

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=interval)

    # Save the animation as a GIF
    ani.save('circle_animation.gif', writer='pillow')

    # Display the GIF
    plt.show()


def multiple_dots_example():
    # CREDIT: ChatGPT4o

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import cm

    # Define the number of frames for the animation
    num_frames = 100

    # Define the number of dots
    num_dots = 5

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # Create dots
    dots = [ax.plot([], [], 'o', markersize=10)[0] for _ in range(num_dots)]

    # Define the parameters x for each dot
    x_values = [np.linspace(-10, 10, num_frames) for _ in range(num_dots)]

    # Initialize the positions of the dots
    initial_angles = np.linspace(0, 2 * np.pi, num_dots, endpoint=False)
    initial_positions = [(np.cos(angle), np.sin(angle)) for angle in initial_angles]

    def get_color(x):
        # Normalize x to be between 0 and 1
        norm_x = (x - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
        # Map the normalized value to a color
        color = cm.coolwarm(norm_x)
        return color

    def init():
        for dot, x in zip(dots, x_values):
            dot.set_data([], [])
            dot.set_color(get_color(x[0]))
        return dots

    def update(frame):
        angles = 2 * np.pi * frame / num_frames + initial_angles
        x_positions = [np.cos(angle) for angle in angles]
        y_positions = [np.sin(angle) for angle in angles]
        for i, dot in enumerate(dots):
            dot.set_data(x_positions[i], y_positions[i])
            dot.set_color(get_color(x_values[i][frame]))
        return dots

    # Adjust the frame rate by setting the interval (in milliseconds)
    frame_rate = 60  # Frames per second
    interval = 1000 // frame_rate

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=interval)

    # Save the animation as a GIF
    ani.save('circle_animation_multiple_dots.gif', writer='pillow')

    # Display the GIF
    plt.show()


def particle_animation():
    # Define the number of frames for the animation
    num_frames = 750

    frame_rate = 100  # Frames per second
    interval = 1000 // frame_rate

    # initialize a grid
    (N1, N2, N3) = (40, 40, 40)
    (xsize, ysize, zsize) = (1., 1., 1.)
    (Dx, Dy, Dz) = (xsize / N1, ysize / N2, zsize / N3)
    g = Grid((N1, N2, N3), (xsize, ysize, zsize), QuadraticSplineFF)

    # initialize some particles with zero total charge
    nparticles = 2
    center = np.array([xsize / 2, ysize / 2, 0])
    R = np.array([0.1, 0., 0.])
    v = np.array([0, 0.19, 0])
    particles = [
        Particle(1.,  0.5, center + R, v=v),  # np.array([0.1, ysize/2, zsize/2])),
        Particle(1., -0.5, center - R, v=-v)  # np.array([xsize/2, 0.1, zsize/2]))
    ]
    qlist = [ptc.q for ptc in particles]  # store the particles' charges

    # pick a time step
    dt = 0.5 * g.max_time_step

    print('Initializing')

    # initialize the simulation
    sim = Simulation(g, particles)
    sim.initialize_charges(dt)
    sim.set_semistatic_init_conds()

    # create a list storing the positions of all particles at various times
    pos_list = [
        [ptc.R.copy() for ptc in particles]
    ]

    def get_field_energy_array():
        #arr = np.zeros((N1, N2))
        #for i in range(N1):
        #    for j in range(N2):
        #        arr[i, j] = np.tanh(2 * (np.linalg.norm(g.E[:, i, j, 0]) ** 2 + np.linalg.norm(g.B[:, i, j, 0]) ** 2))
        #return np.transpose(arr)
        E = g.E[:, :, :, 0]
        B = g.B[:, :, :, 0]
        #return np.transpose(np.tanh(0.5 * np.sum(E**2 + B**2, axis=0)))
        #S = np.array([
        #    E[1] * B[2] - E[2] * B[1],
        #    E[2] * B[0] - E[0] * B[2],
        #    E[0] * B[1] - E[1] * B[2]
        #])
        #return np.transpose(np.tanh(0.5 * np.sum(S**2, axis=0)))
        #return np.transpose(np.log(1 + 3 * np.sum(E**2 + B**2, axis=0)))
        #return np.transpose(np.tanh(np.sqrt(np.sum(E**2 + B**2, axis=0))))
        return np.transpose(np.arctan(1.5 * np.sum(E**2 + B**2, axis=0)))
    field_energy_list = [
        get_field_energy_array()
    ]

    print('Simulating')
    for i in range(num_frames - 1):
        print(i)
        sim.vay_make_step(dt)
        pos_list.append([ptc.R.copy() for ptc in particles])
        field_energy_list.append(get_field_energy_array())

    print('Drawing')

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, xsize)  # LATER CHANGE
    ax.set_ylim(0, ysize)
    ax.set_aspect(xsize / ysize)

    dots = [ax.plot([], [], 'o', markersize=10)[0] for _ in range(nparticles)]

    # Define the two-variable function f(x, y, t) that changes with time t
    def f(xx, yy, t):
        i = round(xx / Dx) % N1
        j = round(yy / Dy) % N2
        #return np.sin(x ** 2 + y ** 2 + t / 20)
        return np.tanh(
            np.linalg.norm(g.E[:, i, j, 0])**2 + np.linalg.norm(g.B[:, i, j, 0])**2
        )
    f = np.vectorize(f)

    x = np.linspace(0, xsize - Dx, N1)
    y = np.linspace(0, ysize - Dy, N2)
    X, Y = np.meshgrid(x, y)
    #Z = f(X, Y, 0)
    Z = field_energy_list[0]
    background = ax.imshow(Z, extent=[0, xsize, 0, ysize], origin='lower', cmap='viridis', alpha=0.5)

    def init():
        for (dot, q) in zip(dots, qlist):
            dot.set_data([], [])
            dot.set_color(cm.coolwarm(0.5 + 0.5 * np.tanh(q)))
        return dots

    def update(frame):
        # Update the background
        #Z = f(X, Y, frame)
        Z = field_energy_list[frame]
        background.set_array(Z)

        x_positions = [R[0] % xsize for R in pos_list[frame]]
        y_positions = [R[1] % ysize for R in pos_list[frame]]
        for i, dot in enumerate(dots):
            dot.set_data(x_positions[i], y_positions[i])
        return dots + [background]

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=interval)

    # Save the animation as a GIF
    ani.save('merger_animation_1.gif', writer='pillow')

    # Display the GIF
    plt.show()


def main():
    #particle_animation()

    np.set_printoptions(suppress=True, precision=2)
    np.random.seed(0)
    random.seed(0)

    unit_test_0()
    unit_test_1()
    unit_test_2()
    unit_test_3()
    unit_test_4()
    unit_test_5()
    unit_test_6()
    unit_test_7()
    unit_test_8()
    unit_test_9()
    unit_test_10()
    unit_test_11()
    unit_test_12()
    unit_test_13()
    unit_test_14()
    unit_test_15()
    unit_test_16()
    unit_test_17()
    unit_test_18()
    unit_test_19()
    unit_test_20()
    unit_test_21()
    unit_test_22()
    unit_test_23()
    unit_test_24()
    unit_test_25()

    #intg_test_0()
    #intg_test_1()
    #intg_test_2()
    #intg_test_3()
    #intg_test_4()
    #intg_test_5()
    #intg_test_6()
    #intg_test_7()

    #soft_test_0()
    #soft_test_1()
    #soft_test_2()
    #soft_test_3()
    soft_test_4()

    input('Press ENTER to complete')


if __name__ == '__main__':
    main()
