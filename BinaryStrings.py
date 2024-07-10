import random

import numpy as np
import itertools


# check if x is one of several kinds of integers
def is_int(x):
    return isinstance(x, int) or isinstance(x, np.int32) or isinstance(x, np.int64) or isinstance(x, np.intc)


# given a binary string b_0...b_{b-1}, represented by a nonnegative integer
# b = b_0 + 2 b_1 + ... + 2^{n-1} b_{n-1}, compute the
# "weight" b_0 + b_1 + ... + b_{n-1}.
def binary_weight(b):
    assert is_int(b) and b >= 0
    return bin(b).count('1')


# given two binary strings b_0...b_{n-1} and c_0...c_{n-1}, represented by nonnegative integers
# b = b_0 + 2 b_1 + ... + 2^{n-1} b_{n-1} and c = (...), compute their
# "dot product" b_0 c_0 + b_1 c_1 + ... + b_{n-1} c_{n-1}.
def binary_dot(b, c):
    assert is_int(b) and is_int(c) and b >= 0 and c >= 0
    return binary_weight(np.bitwise_and(b, c))


# given a binary string b_0...b_{b-1}, represented by a nonnegative integer
# b = b_0 + 2 b_1 + ... + 2^{n-1} b_{n-1},
# return b_i.
def binary_string_at(b, i):
    return np.right_shift(b, i) % 2


# given a numpy array [b_0, ..., b_{n-1}],
# compute b_0 + 2 b_1 + ... + 2^{n-1} b_{n-1}.
# CREDIT: ChatGPT 4o
def binary_to_int(arr):
    n = arr.size
    powers_of_2 = 2 ** np.arange(n)

    return np.sum(arr * powers_of_2)


# given an n x n full-rank matrix A and an n x 1 vector b in the field {0, 1},
# find an n-component vector x in the field {0, 1} such that A x = b.
# note: A is represented as the numpy array of its rows [r_1, ..., r_n],
# and b, r_1, ..., r_n are represented by nonnegative integers as in binary_weight.
def f2_linear_solve(n, A_mat, b):
    A = A_mat.copy()

    assert isinstance(n, int) and n > 0  # the size of the linear system
    assert isinstance(b, int) and b >= 0
    assert isinstance(A, np.ndarray) and A.shape == (n,)
    assert all(map(lambda r: is_int(r) and r >= 0, A))

    # implement Gauss-Jordan elimination over the {0, 1} finite field
    for i in range(n):
        # find positions of nonzero entries in the i-th column of A
        col = np.bitwise_and(np.right_shift(A, i), 1)
        potential_pivots = list(np.nonzero(col)[0])
        potential_pivots = list(filter(lambda j: j >= i, potential_pivots))

        if not potential_pivots:
            raise ValueError  # A is not invertible

        piv = potential_pivots[0]
        if piv != i:
            # swap rows
            A[i], A[piv] = A[piv], A[i]

            val1, val2 = binary_string_at(b, i), binary_string_at(b, piv)
            mask1, mask2 = 2 ** i, 2 ** piv
            mask3 = np.bitwise_not(np.bitwise_or(mask1, mask2))
            b = np.bitwise_and(b, mask3) + mask2 * val1 + mask1 * val2

        # create an array whose j-th entry will be one if row i should be added to row j and zero otherwise
        incr = np.bitwise_and(np.right_shift(A, i), 1)
        incr[i] = 0

        # make the i-th row of A 0...010...0 and modify b accordingly
        delta_A = incr * A[i]
        A = np.bitwise_xor(A, delta_A)  # addition mod 2 over n bits at a time
        delta_b = binary_string_at(b, i) * binary_to_int(incr)
        b = np.bitwise_xor(b, delta_b)  # addition mod 2 over n bits at a time

    return b


# given binary strings init_regs = [r_0, ..., r_{n-1}], specified by integers as in binary_to_int,
# and a sequence of xor/cnot operations (i, j) mapping r_i -> r_i, r_j -> bitwise_xor(r_i, r_j),
# compute the result of applying this sequence of operations to the registers.
def simulate_xor_seq(init_regs, seq):
    assert isinstance(init_regs, np.ndarray)
    assert all(map(lambda x: isinstance(x, np.int32), init_regs))
    regs = init_regs.copy()
    n = len(regs)

    for move in seq:
        i, j = move

        assert isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64)
        assert isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64)
        assert i != j
        assert 0 <= i < n and 0 <= j < n

        regs[j] = np.bitwise_xor(regs[i], regs[j])

    return regs


# return a list of tuples (i, j), |i - j|= 1, specifying xor/cnot operations mapping r_j to bitwise_xor(r_i, r_j)
# such that r_{start + length} is mapped to bitwise_xor(r_start, ..., r_{start + length}), and the
# registers r_{start + 1}, ..., r_{start + length - 1} are mapped to some OTHER VALUES .
def xor_accumulate_down(start, length):
    return [(i, i + 1) for i in range(start, start + length)]


# return a list of tuples (i, j), |i - j|= 1, specifying xor/cnot operations mapping r_j to bitwise_xor(r_i, r_j)
# such that r_{start - length} is mapped to bitwise_xor(r_{start - length}, ..., r_start), and the
# registers r_{start - length + 1}, ..., r_{start - 1} are mapped to some OTHER VALUES .
def xor_accumulate_up(start, length):
    return [(i, i - 1) for i in range(start, start - length, -1)]


# return a list of tuples (i, j), |i - j|= 1, specifying xor/cnot operations mapping r_j to bitwise_xor(r_i, r_j)
# such that r_{start + length} is mapped to r_start, and the registers
# r_{start + 1}, ..., r_{start + length - 1} are mapped to some OTHER VALUES .
def xor_skip_down(start, length):
    result = [(i, i - 1) for i in range(start + length, start, -1)]
    result += [(start, start + 1)]
    for i in range(start + 1, start + length):
        result += [(i + 1, i), (i, i + 1)]
    return result


# return a list of tuples (i, j), |i - j|= 1, specifying xor/cnot operations mapping r_j to bitwise_xor(r_i, r_j)
# such that r_{start - length} is mapped to r_start, and the registers
# r_{start - length + 1}, ..., r_{start - 1} are mapped to some OTHER VALUES .
def xor_skip_up(start, length):
    result = [(i, i+1) for i in range(start - length, start)]
    result += [(start, start - 1)]
    for i in range(start - 1, start - length, -1):
        result += [(i - 1, i), (i, i - 1)]
    return result


# return a sequence of tuples (i, j) specifying xor operations (see simulate_xor_seq) such that
# register r_{start + length - 1} is set to
# bitwise_xor(b_0 r_start, b_1 r_{start + 1}, ..., b_{length - 1} r_{start + length - 1}),
# where b is specified by an integer b_0 + 2 b_1 + ... + 2^{length-1} b_{length-1}.
def xor_combination_to_lower(start, length, b):
    assert isinstance(b, int) or isinstance(b, np.int32) or isinstance(b, np.int64)
    assert b != 0  # cannot make the register be just an empty value
    assert isinstance(start, int) and start >= 0
    assert isinstance(length, int) and length >= 1

    # compute [b_0, ..., b_{length-1}]
    b_arr = np.zeros(length)
    for i in range(length):
        b_arr[i] = b % 2
        b = np.right_shift(b, 1)

    # place xor_accumulate_down and xor_skip_down as appropriate
    first_nonzero = np.nonzero(b_arr)[0][0]  # smallest i such that b_i == 1
    if first_nonzero == length - 1:
        # if b == 0...01, the register already holds the desired value
        return []
    else:
        seq = []  # this will store the sequence of xor/cnot
        curr_pos = start + first_nonzero  # index of the current "active" register

        # compute the part of b not considered so far and split it into runs of zeros and ones
        remaining_b_arr = b_arr[first_nonzero + 1:]  # b_{i+1} ... b_{length-1}
        consecutive_runs = [(label, sum(1 for _ in group)) for label, group in itertools.groupby(remaining_b_arr)]

        for char, count in consecutive_runs:
            if char == 0:
                # skip count positions down
                seq += xor_skip_down(curr_pos, count)
                curr_pos += count
            else:  # char == 1
                # accumulate count positions down
                seq += xor_accumulate_down(curr_pos, count)
                curr_pos += count

        return seq


# return a sequence of tuples (i, j) specifying xor operations (see simulate_xor_seq) such that
# register r_start is set to
# bitwise_xor(b_0 r_{start + length - 1}, b_1 r_{start + length - 2}, ..., b_{length - 1} r_start),
# where b is specified by an integer b_0 + 2 b_1 + ... + 2^{length-1} b_{length-1}.
def xor_combination_to_upper(start, length, b):
    assert isinstance(b, int) or isinstance(b, np.int32) or isinstance(b, np.int64)
    assert b != 0  # cannot make the register be just an empty value
    assert isinstance(start, int) and start >= 0
    assert isinstance(length, int) and length >= 1

    # compute [b_0, ..., b_{length-1}]
    b_arr = np.zeros(length)
    for i in range(length):
        b_arr[i] = b % 2
        b = np.right_shift(b, 1)

    # place xor_accumulate_up and xor_skip_up as appropriate
    last_nonzero = np.nonzero(b_arr)[0][-1]  # largest i such that b_i == 1
    if last_nonzero == 0:
        # if b == 10..0, the register already holds the desired value
        return []
    else:
        seq = []  # this will store the sequence of xor/cnot
        curr_pos = start + last_nonzero  # index of the current "active" register

        # compute the part of b not considered so far and split it into runs of zeros and ones
        remaining_b_arr = reversed(b_arr[:last_nonzero])  # b_{i-1} ... b_0
        consecutive_runs = [(label, sum(1 for _ in group)) for label, group in itertools.groupby(remaining_b_arr)]

        for char, count in consecutive_runs:
            if char == 0:
                # skip count positions up
                seq += xor_skip_up(curr_pos, count)
                curr_pos -= count
            else:  # char == 1
                # accumulate count positions up
                seq += xor_accumulate_up(curr_pos, count)
                curr_pos -= count

        return seq


# return a sequence of tuples (i, j) specifying xor operations (see simulate_xor_seq) such that
# register r_{lower + m} is set to
# bitwise_xor(b_0 r_lower, b_1 r_{lower + 1}, ..., b_{length - 1} r_{lower + length - 1}),
# where b is specified by an integer b_0 + 2 b_1 + ... + 2^{length-1} b_{length-1}.
def adjacent_xor_sequence(lower, length, m, b):
    assert isinstance(b, int) or isinstance(b, np.int32) or isinstance(b, np.int64)  # goal b_0 b_1 ... b_{length-1}
    assert isinstance(m, int) and 0 <= m < length  # lower + m is the register in which the goal value will be written
    assert length >= 1

    if length == 1:
        assert b == 1  # the problem is unsolvable otherwise
        # if length == 1, then m == 0
        return []  # the register already holds the desired value

    # goal = g_lower g_middl g_upper
    g_upper = np.bitwise_and(b, 2**m - 1)  # the first m bits of goal
    g_middl = binary_string_at(b, m)       # the reg-th bit of goal
    g_lower = np.right_shift(b, m + 1)     # all bits after position m + 1

    # sequence to place g_upper into register lower + m - 1
    if g_upper:
        seq_upper = xor_combination_to_lower(lower, m, g_upper)
    else:
        seq_upper = []

    # sequence to place g_lower into register lower + m + 1
    if g_lower:
        seq_lower = xor_combination_to_upper(lower + m + 1, length - m - 1, g_lower)
    else:
        seq_lower = []

    # combine upper, middle, and lower parts
    if g_middl:  # if data from lower + m is included
        add_dn = [(lower + m + 1, lower + m)]
        add_up = [(lower + m - 1, lower + m)]

        if m == 0:  # when the target register is on the upper edge
            if g_lower:
                # if there is anything to add from below
                return seq_lower + add_dn
            else:
                return []
        elif m == length - 1:  # when the target register is on the lower edge
            if g_upper:
                # if there is anything to add from above
                return seq_upper + add_up
            else:
                return []
        else:  # 1 <= m <= length - 2, which is the middle case
            seq = []
            if g_lower:
                seq += seq_lower + add_dn
            if g_upper:
                seq += seq_upper + add_up
            return seq
    else:  # if data from lower + m is not included, have to move it elsewhere
        move_dn_and_add = [(lower + m, lower + m + 1), (lower + m + 1, lower + m)]
        move_up_and_add = [(lower + m, lower + m - 1), (lower + m - 1, lower + m)]

        add_dn = [(lower + m + 1, lower + m)]
        add_up = [(lower + m - 1, lower + m)]

        if m == 0:
            # the only option is to move down.
            # note that since g_middl == g_upper == 0, it has to b that g_lower is nonempty.
            return seq_lower + move_dn_and_add
        elif m == length - 1:
            # the only option is to move up.
            # note that since g_middl == g_lower == 0, it has to be that g_upper is nonempty.
            return seq_upper + move_up_and_add
        else:  # 1 <= m <= length - 2
            # have the option to move up or down
            if g_lower:
                seq = seq_lower + move_dn_and_add
                if g_upper:
                    seq += seq_upper + add_up
                return seq
            else:
                # have what to add above but not below.
                # note that since g_middl == g_lower == 0, it has to be that g_upper is nonempty.
                return seq_upper + move_up_and_add


# given a binary-valued matrix A_{ij} specified by a list of its rows, each row encoded as an integer,
# output the transpose of A, encoded by a list of its rows, each row encoded as an integer.
def transpose_bin_matrix(A):
    n = len(A)

    return np.array([binary_to_int(np.bitwise_and(np.right_shift(A, i), 1)) for i in range(n)])


# given regs = [r_0, ..., r_{n - 1}], output a sequence of tuples (i, j), with |i - j| = 1,
# and an integer m, specifying xor operations (see simulate_xor_seq)
# such that r_m becomes equal to b.
# note: only looking at characters 0 ... n-1 of each register.
def make_desired_sting(regs, b):
    n = len(regs)
    A = transpose_bin_matrix(regs)
    lin_comb = f2_linear_solve(n, A, b)

    m = 0
    seq = adjacent_xor_sequence(0, n, 0, lin_comb)
    for m_candidate in range(1, n):
        seq_candidate = adjacent_xor_sequence(0, n, m_candidate, lin_comb)
        if len(seq_candidate) < len(seq):
            seq = seq_candidate
            m = m_candidate

    return seq, m


# compute a solution to the string breeding problem on n registers by always making the closest goal string.
# NOTE: this is a RANDOMIZED algorithm, since "ties" between equally close strings are resolved randomly.
def sbp_solve_greedy(n):
    # set goals to be the set of all binary strings of length n and odd weight, except those that have weight 1
    goals = set()
    for b in range(2**n):
        w = binary_weight(b)
        if w % 2 == 1 and w > 1:
            goals.add(b)

    # initialize the register values
    regs = np.array([2 ** i for i in range(n)])

    # initialize a variable to store the sequence of cnot and z-rotation gate specifications
    output = []

    # take the goals on by one, always choosing to pursue the nearest
    while goals:
        best_options = []
        best_distance = 999 * n**2  # "infinity"
        for g in goals:
            seq, m = make_desired_sting(regs, g)
            option = (seq, m, g)
            l = len(seq)
            if l < best_distance:
                best_distance = l
                best_options = [option]
            elif l == best_distance:
                best_options.append(option)

        # randomly choose one of the closest goals.
        # record that the goal has been visited, update register values, and
        # record the sequence used to visit this goal.
        seq, m, g = random.choice(best_options)
        goals.remove(g)
        regs = simulate_xor_seq(regs, seq)
        output += seq + [('R', g, m)]  # "rotate register m, which holds binary string g"

    return output


# compute the circuit depth of a given "string breeding problem" solution for n qubits,
# generated, e.g., by sbp_solve_greedy
def get_sbp_soln_depth(n, soln):
    depths = np.zeros(n, dtype=int)
    for move in soln:
        if len(move) == 2:  # a CNOT gate
            (i, j) = move
            m = max(depths[i], depths[j]) + 1
            depths[i] = depths[j] = m
        else:  # a rotation gate
            _, _, j = move
            depths[j] += 1
    return max(depths)


# randomly generate a sequence of k tuples (i, j), with |i - j| = 1 and start <= i, j < start + length
def random_xor_seq(start, length, k):
    out = []

    for iteration in range(k):
        u = random.randint(start, start + length - 2)
        if random.getrandbits(1):  # cnot gate going down
            (i, j) = (u, u + 1)
        else:  # cnot gate going up
            (i, j) = (u + 1, u)
        out.append((i, j))

    return out
