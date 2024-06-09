import numpy as np
import random

# -------------------------------------------------------------------------------------
#                                 FORM FACTOR
# -------------------------------------------------------------------------------------
# A small class to represent and evaluate a particle form factor. Does not rely on
# any other class.
#
# POSSIBLE ADDITIONS:
#   [*] Implement new form factors in addition to the quadratic spline form factor
#             (which is not hard because it does not modify the class itself)


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


# a bell-shaped particle of diameter 3
QuadraticSplineFF = FormFactor(lambda x, y, z: one_dim_quadr_spline(x) *
                                               one_dim_quadr_spline(y) *
                                               one_dim_quadr_spline(z), 1)

# to be used as a PLACEHOLDER only; not a physically meaningful form factor
EmptyFF = FormFactor(None, 0)
