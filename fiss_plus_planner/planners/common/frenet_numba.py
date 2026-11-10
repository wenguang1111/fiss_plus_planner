"""Numba versions of FrenetOptimalPlanner.calc_frenet_paths / calc_global_paths.

They reproduce the Python reference bit for bit: every expression keeps the reference's operation order,
and every power goes through libm pow() like the reference's numpy-scalar `**` and np.power do. LLVM
rewrites a constant `x ** 2.0` into `x * x`, which differs from pow() in the last bit for ~0.1% of inputs,
so the exponents are passed in as runtime values (_POWERS) instead of literals.
"""
import math

import numpy as np
from numba import njit

from fiss_plus_planner.planners.common.scenario.frenet import FrenetTrajectory, SamplingParam

_POWERS = np.array([2.0, 3.0, 4.0, 5.0])


@njit(cache=True)
def _eval_frenet_paths(t, lat, lon, pw, w_T, w_V, w_A, w_J, w_LC, max_speed):
    """t: (L,) time samples; lat: (N, 6) quintic coeffs of d(t); lon: (N, 5) quartic coeffs of s(t).

    Returns out (N, 8, L) = d, d_d, d_dd, d_ddd, s, s_d, s_dd, s_ddd and cost (N,) = CostFunction.cost_total.
    """
    two, three, four, five = pw[0], pw[1], pw[2], pw[3]
    n_traj, n_pts = lat.shape[0], t.shape[0]
    out = np.empty((n_traj, 8, n_pts))
    cost = np.empty(n_traj)
    for n in range(n_traj):
        a0, a1, a2, a3, a4, a5 = lat[n, 0], lat[n, 1], lat[n, 2], lat[n, 3], lat[n, 4], lat[n, 5]
        b0, b1, b2, b3, b4 = lon[n, 0], lon[n, 1], lon[n, 2], lon[n, 3], lon[n, 4]
        for k in range(n_pts):
            tk = t[k]
            t2 = tk ** two
            t3 = tk ** three
            t4 = tk ** four
            t5 = tk ** five
            out[n, 0, k] = a0 + a1 * tk + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
            out[n, 1, k] = a1 + 2 * a2 * tk + 3 * a3 * t2 + 4 * a4 * t3 + 5 * a5 * t4
            out[n, 2, k] = 2 * a2 + 6 * a3 * tk + 12 * a4 * t2 + 20 * a5 * t3
            out[n, 3, k] = 6 * a3 + 24 * a4 * tk + 60 * a5 * t2
            out[n, 4, k] = b0 + b1 * tk + b2 * t2 + b3 * t3 + b4 * t4
            out[n, 5, k] = b1 + 2 * b2 * tk + 3 * b3 * t2 + 4 * b4 * t3
            out[n, 6, k] = 2 * b2 + 6 * b3 * tk + 12 * b4 * t2
            out[n, 7, k] = 6 * b3 + 24 * b4 * tk

        # CostFunction.cost_total; the sums are sequential like the builtin sum() the reference uses.
        sum_speed = 0.0
        sum_s_dd = 0.0
        sum_d_dd = 0.0
        sum_s_ddd = 0.0
        sum_d_ddd = 0.0
        sum_d = 0.0
        for k in range(n_pts):
            sum_speed += (abs(out[n, 5, k]) - max_speed) ** two
            sum_s_dd += out[n, 6, k] ** two
            sum_d_dd += out[n, 2, k] ** two
            sum_s_ddd += out[n, 7, k] ** two
            sum_d_ddd += out[n, 3, k] ** two
            sum_d += out[n, 0, k] ** two
        cost_time = w_T * (15.0 - 0.1 * n_pts)
        cost_speed = w_V * sum_speed
        cost_accel = w_A * sum_s_dd + w_A * sum_d_dd
        cost_jerk = w_J * sum_s_ddd + w_J * sum_d_ddd
        cost_offset = w_LC * sum_d
        cost[n] = (cost_time + 0.0 + cost_speed + cost_accel + cost_jerk + cost_offset) / n_pts
    return out, cost


@njit(cache=True)
def _eval_global_paths(s, d, knots, ax, bx, cx, dx, ay, by, cy, dy, pw):
    """Map Frenet (s, d) samples (N, L) to global (x, y) along the CubicSpline2D given by its coefficients.

    Like the reference, a trajectory stops at its first point outside the spline; n_valid holds the count.
    """
    two, three = pw[0], pw[1]
    n_traj, n_pts = s.shape
    x = np.empty((n_traj, n_pts))
    y = np.empty((n_traj, n_pts))
    n_valid = np.zeros(n_traj, dtype=np.int64)
    s_min, s_max = knots[0], knots[knots.shape[0] - 1]
    half_pi = math.pi / 2.0
    for n in range(n_traj):
        for k in range(n_pts):
            sv = s[n, k]
            if sv < s_min or sv > s_max:
                break
            i = np.searchsorted(knots, sv, side="right") - 1  # == bisect.bisect(knots, sv) - 1
            if i >= bx.shape[0]:
                # The reference indexes b[i] past its end here (sv == last knot) and raises too.
                raise IndexError("arc length at the last spline knot")
            h = sv - knots[i]
            h2 = h ** two
            h3 = h ** three
            ix = ax[i] + bx[i] * h + cx[i] * h2 + dx[i] * h3
            iy = ay[i] + by[i] * h + cy[i] * h2 + dy[i] * h3
            yaw = math.atan2(by[i] + 2.0 * cy[i] * h + 3.0 * dy[i] * h2,
                             bx[i] + 2.0 * cx[i] * h + 3.0 * dx[i] * h2)
            di = d[n, k]
            x[n, k] = ix + di * math.cos(yaw + half_pi)
            y[n, k] = iy + di * math.sin(yaw + half_pi)
            n_valid[n] += 1
    return x, y, n_valid


def _quintic_coeffs(xs, vxs, axs, xe, vxe, axe, time):
    """QuinticPolynomial for an array of end positions xe sharing one time; returns (N, 6)."""
    a0, a1, a2 = xs, vxs, axs / 2.0
    A = np.array([[time ** 3, time ** 4, time ** 5],
                  [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                  [6 * time, 12 * time ** 2, 20 * time ** 3]])
    b = np.empty((len(xe), 3))
    b[:, 0] = xe - a0 - a1 * time - a2 * time ** 2
    b[:, 1] = vxe - a1 - 2 * a2 * time
    b[:, 2] = axe - 2 * a2
    sol = np.linalg.solve(np.broadcast_to(A, (len(xe), 3, 3)), b)
    coeffs = np.empty((len(xe), 6))
    coeffs[:, 0], coeffs[:, 1], coeffs[:, 2] = a0, a1, a2
    coeffs[:, 3:] = sol
    return coeffs


def _quartic_coeffs(xs, vxs, axs, vxe, axe, time):
    """QuarticPolynomial for an array of end speeds vxe sharing one time; returns (N, 5)."""
    a0, a1, a2 = xs, vxs, axs / 2.0
    A = np.array([[3 * time ** 2, 4 * time ** 3],
                  [6 * time, 12 * time ** 2]])
    b = np.empty((len(vxe), 2))
    b[:, 0] = vxe - a1 - 2 * a2 * time
    b[:, 1] = axe - 2 * a2
    sol = np.linalg.solve(np.broadcast_to(A, (len(vxe), 2, 2)), b)
    coeffs = np.empty((len(vxe), 5))
    coeffs[:, 0], coeffs[:, 1], coeffs[:, 2] = a0, a1, a2
    coeffs[:, 3:] = sol
    return coeffs


def calc_frenet_paths(planner, frenet_state, samples) -> list:
    """Drop-in for FrenetOptimalPlanner.calc_frenet_paths; trajectory fields are numpy arrays."""
    settings, cost_fn = planner.settings, planner.cost_function
    if len(samples) == 0:
        return []
    samples = np.asarray(samples, dtype=float)
    di_all, tv_all = samples[:, 0], samples[:, 1]
    ti_all = np.maximum(samples[:, 2], settings.tick_t)  # same clamp as the reference

    paths = [None] * len(samples)
    for ti in np.unique(ti_all):
        idx = np.flatnonzero(ti_all == ti)
        t = np.arange(0.0, ti, settings.tick_t)
        lat = _quintic_coeffs(frenet_state.d, frenet_state.d_d, frenet_state.d_dd, di_all[idx], 0.0, 0.0, ti)
        lon = _quartic_coeffs(frenet_state.s, frenet_state.s_d, frenet_state.s_dd, tv_all[idx], 0.0, ti)
        out, cost = _eval_frenet_paths(t, lat, lon, _POWERS, cost_fn.w_T, cost_fn.w_V, cost_fn.w_A,
                                       cost_fn.w_J, cost_fn.w_LC, cost_fn.max_speed)
        for row, n in enumerate(idx):
            fp = FrenetTrajectory()
            fp.t = t.copy()
            fp.d, fp.d_d, fp.d_dd, fp.d_ddd = out[row, 0], out[row, 1], out[row, 2], out[row, 3]
            fp.s, fp.s_d, fp.s_dd, fp.s_ddd = out[row, 4], out[row, 5], out[row, 6], out[row, 7]
            fp.sampling_param = SamplingParam(di_all[n], tv_all[n], ti)
            fp.cost_final = cost[row]
            paths[n] = fp

    planner.all_trajs.append(list(paths))
    return paths


def spline_coefficients(cubic_spline):
    """Flatten a CubicSpline2D into the arrays _eval_global_paths takes."""
    sx, sy = cubic_spline.sx, cubic_spline.sy
    as_array = lambda values: np.ascontiguousarray(values, dtype=float)
    return (as_array(sx.x), as_array(sx.a), as_array(sx.b), as_array(sx.c), as_array(sx.d),
            as_array(sy.a), as_array(sy.b), as_array(sy.c), as_array(sy.d))


def calc_global_paths(planner, fplist: list, spline_coeffs) -> list:
    """Drop-in for FrenetOptimalPlanner.calc_global_paths, given spline_coefficients(planner.cubic_spline)."""
    if not fplist:
        return []
    dt = planner.settings.tick_t
    keep = [False] * len(fplist)
    # Trajectories of one length are mapped together; with a fixed horizon that is all of them.
    for idx in _group_by(range(len(fplist)), lambda i: len(fplist[i].s)):
        s = np.stack([fplist[i].s for i in idx])
        d = np.stack([fplist[i].d for i in idx])
        x, y, n_valid = _eval_global_paths(s, d, *spline_coeffs, _POWERS)
        # Same element-wise numpy ops as the reference, on all trajectories of one valid length at once.
        for rows in _group_by(range(len(idx)), lambda r: n_valid[r]):
            m = n_valid[rows[0]]
            if m < 2:
                continue
            rows = np.asarray(rows)
            gx, gy = x[rows, :m], y[rows, :m]
            x_d = np.diff(gx, axis=1)
            y_d = np.diff(gy, axis=1)
            yaw = np.arctan2(y_d, x_d)
            ds = np.hypot(x_d, y_d)
            yaw = np.concatenate((yaw, yaw[:, -1:]), axis=1)
            c = np.divide(np.diff(yaw, axis=1), ds)
            c_d = np.divide(np.diff(c, axis=1), dt)
            c_dd = np.divide(np.diff(c_d, axis=1), dt)
            for j, r in enumerate(rows):
                fp = fplist[idx[r]]
                fp.x, fp.y, fp.yaw, fp.ds = gx[j], gy[j], yaw[j], ds[j]
                fp.c, fp.c_d, fp.c_dd = c[j], c_d[j], c_dd[j]
                keep[idx[r]] = True
    return [fp for fp, k in zip(fplist, keep) if k]


def _group_by(items, key) -> list:
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return list(groups.values())
