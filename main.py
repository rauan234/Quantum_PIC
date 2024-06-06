import numpy as np
import random

from Grid import *
from Particle import *
from FormFactor import *
from tests import *


def main():
    np.set_printoptions(suppress=True, precision=3)
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

    intg_test_0()
    intg_test_1()

    #soft_test_0()
    #soft_test_1()
    #soft_test_2()

    #input('Press ENTER to complete')


if __name__ == '__main__':
    main()
