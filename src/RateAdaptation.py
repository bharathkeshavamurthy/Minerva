import math
import traceback
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import rice, ncx2, rayleigh


def __f_z(z):
    return 160e3 * k_su * math.log2(1 + (0.5 * math.pow(z, 2)))


# noinspection PyMethodMayBeStatic
def __marcum_q(df, nc, x):
    return 1 - ncx2.cdf(x, df, nc)


def __f(z, *args):
    df, nc, y = args
    f_z, q_m = __f_z(z), __marcum_q(df, nc, (y * (z ** 2)))
    ln_f_z = math.log(f_z) if f_z != 0.0 else np.inf
    ln_q_m = math.log(q_m) if q_m != 0.0 else np.inf
    return (-1 * ln_f_z) - ln_q_m


# noinspection PyMethodMayBeStatic
def __bisect(f, df, nc, y, low, high, tolerance):
    args = (df, nc, y)
    assert tolerance is not None
    mid, converged, conf, conf_th = 0.0, False, 0, 10
    while (not converged) or (conf < conf_th):
        mid = (high + low) / 2
        if (f(low, *args) * f(high, *args)) > 0.0:
            low = mid
        else:
            high = mid
        converged = (abs(high - low) < tolerance)
        conf += 1 if converged else (-1 * conf)  # Update OR Reset if it is a red herring
    return mid


def __z(gamma):
    return math.sqrt(2 * (math.pow((gamma / (160e3 * k_su)), 2) - 1.0))


def __u(gamma, d, los):
    return (math.pow(2, (gamma / (160e3 * k_su))) - 1) / ((160e3 * k_su) * (d ** (-1 * (lambda: 2.8,
                                                                                        lambda: 2.0)[los]())))


def __evaluate_los_throughput(p, d, phi, r_bar, adapt=True):
    k = 1.0 * math.exp(0.0512 * phi)
    df, nc = 2, (2 * k)
    y = (k + 1) * (1 / (1e5 * (d ** (-1 * 2.0))))
    z_star = __bisect(__f, df, nc, y, 0,
                      __z(((160e3 * k_su) * math.log2(1 + (math.pow(rice.ppf(0.9999, k), 2) * 1e5 *
                                                           (d ** (-1 * 2.0))))) + 100.0), 1e-6)
    gamma_star = __f_z(z_star)
    x_star = 2 * (k + 1) * __u(gamma_star, d, True)
    r_star = gamma_star * __marcum_q(df, nc, x_star)
    try:
        tf.compat.v1.assign(r_bar, r_bar + (p * (r_star if adapt else 9e5)), validate_shape=True, use_locking=True)
    except Exception as e__:
        print('[ERROR] SMDPEvaluation __evaluate_los_throughput: Exception caught during tensor assignment - '
              '{}'.format(traceback.print_tb(e__.__traceback__)))
    # Nothing to return...


def __evaluate_nlos_throughput(p, d, r_bar, adapt=True):
    df, nc = 2, 0
    y = 1 / (1e5 * (0.2 * d ** (-1 * 2.8)))
    z_star = __bisect(__f, df, nc, y, 0,
                      __z(((160e3 * k_su) * math.log2(1 + (math.pow(rayleigh.ppf(0.9999), 2) *
                                                           1e5 * (0.2 * d ** (-1 * 2.8))))) + 100.0), 1e-6)
    gamma_star = __f_z(z_star)
    x_star = 2 * __u(gamma_star, d, False)
    r_star = gamma_star * __marcum_q(df, nc, x_star)
    # Try-Catch Block to handle resource access exceptions
    try:
        tf.compat.v1.assign(r_bar, r_bar + (p * (r_star if adapt else 9e5)), validate_shape=True, use_locking=True)
    except Exception as e__:
        print('[ERROR] [{}] SMDPEvaluation __evaluate_los_throughput: Exception caught during tensor assignment - '
              '{}'.format(id, traceback.print_tb(e__.__traceback__)))
    # Nothing to return...


def __calculate_adapted_throughput(d, phi, r_bar, adapt=True):
    p = 1 / (1 + (9.12 * math.exp(-1 * 0.16 * (phi - 9.12))))
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(__evaluate_los_throughput, p, d, phi, r_bar, adapt)
        executor.submit(__evaluate_nlos_throughput, (1 - p), d, r_bar, adapt)
    # Nothing to return...


# c_pus = [0.9, 0.9, 0.860625, 0.826875, 0.81, 0.7875, 0.781875, 0.73125, 0.691875, 0.6525, 0.61875, 0.6046875,
#          0.45, 0.36, 0.27]
# for c_pu in c_pus:
#     k_su = (c_pu * 6) / 0.9
#     r_var = tf.Variable(0.0, dtype=tf.float64)
#     __calculate_adapted_throughput(10, 0, r_var, False)
#     print(r_var / (6 * 1e6))

# c_sus = [0, 0.774, 1.31, 1.84, 2.59, 3.04, 3.47, 4, 4.44, 4.77, 5.55, 6.33, 7.1, 7.4, 8.15]
# c_sus = [0, 0.75, 1.35, 2.1, 2.55, 3, 3.3, 3.9, 4.5, 4.95, 5.5, 6.6, 7.275, 7.68, 8.37]
# for c_su in c_sus:
#     k_su = c_su / 0.6
#     r_var = tf.Variable(0.0, dtype=tf.float64)
#     __calculate_adapted_throughput(100, 45, r_var)
#     print(r_var / 1e6)

# ROC Evaluation
k_su, data = 0, dict()
c_sus = np.array([0.0, 0.75, 1.35, 2.1, 2.55, 3.0, 3.3, 3.9, 4.5, 4.95, 5.5, 6.6, 7.275, 7.68, 8.37]) / 0.6
c_pus = (np.array([0.9, 0.9, 0.860625, 0.826875, 0.81, 0.7875, 0.781875, 0.73125, 0.691875, 0.6525, 0.61875, 0.6046875,
                   0.45, 0.36, 0.27, 0.0]) * 6) / 0.9
max_occupancies = max(c_pus)
p_mds = (max_occupancies - c_pus) / max_occupancies
p_fas = np.array([((13.95 - k) / 13.95) for k in c_sus])
print(p_mds)
print(p_fas)
