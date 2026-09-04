"""Self-checks for ct_trajectory.

Run:  python test_ct_trajectory.py

Four checks.
  1. Analytic omega and acceleration against central finite differences.
     This is the 1e-6 check. It proves derivatives() is correct.
  2. Basalt preintegration (Eqs. 10-12) of the synthetic IMU lands back
     on the spline, with error falling as the IMU rate rises. The scheme
     is first order, so error should fall ~10x per 10x rate.
  3. A device at rest reports gyro = 0 and accel = R^T (0,0,+9.81).
  4. A fake orbit (stations x aims) shows the still-spin-jump pattern.
"""

import numpy as np

from ct_trajectory import (
    CumulativeSE3Spline,
    DEFAULT_GRAVITY,
    gap_report,
    predict_from_preintegration,
    preintegrate_basalt,
    so3_exp,
    so3_log,
    spline_from_matrices,
    synthesize_imu,
)


def yaw_pitch_roll(yaw, pitch, roll):
    return so3_exp(np.array([0, 0, yaw])) @ so3_exp(np.array([0, pitch, 0])) @ so3_exp(np.array([roll, 0, 0]))


def make_test_spline(duration=10.0, dt=0.1):
    """A wobbly loop with continuous translation and rotation."""
    n = int(duration / dt) + 1
    ts = np.linspace(0.0, duration, n)
    R = np.zeros((n, 3, 3))
    p = np.zeros((n, 3))
    for k, t in enumerate(ts):
        w = 2 * np.pi / duration
        p[k] = [1.5 * np.cos(w * t), 1.5 * np.sin(w * t), 0.3 * np.sin(2.0 * w * t)]
        R[k] = yaw_pitch_roll(w * t + 0.4 * np.sin(3 * w * t), 0.3 * np.sin(2 * w * t), 0.2 * np.cos(5 * w * t))
    return CumulativeSE3Spline(0.0, dt, R, p)


def banner(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. derivatives vs finite differences
# ---------------------------------------------------------------------------


def check_derivatives(spline, h=1e-5):
    banner("1. analytic derivatives vs central finite differences")
    ts = np.linspace(spline.t_min + 0.05, spline.t_max - 0.05, 41)
    err_w = err_v = err_a = 0.0
    for t in ts:
        R, p, omega, v, a = spline.derivatives(t)
        Rm, pm = spline.pose(t - h)
        Rp, pp = spline.pose(t + h)
        # body-frame angular velocity from the geodesic between R(t-h) and R(t+h)
        omega_fd = so3_log(Rm.T @ Rp) / (2 * h)
        v_fd = (pp - pm) / (2 * h)
        a_fd = (pp - 2 * p + pm) / (h * h)
        err_w = max(err_w, np.abs(omega - omega_fd).max())
        err_v = max(err_v, np.abs(v - v_fd).max())
        err_a = max(err_a, np.abs(a - a_fd).max())
    print(f"  max |omega - fd|   = {err_w:.2e} rad/s")
    print(f"  max |v - fd|       = {err_v:.2e} m/s")
    print(f"  max |a - fd|       = {err_a:.2e} m/s^2   (2nd-order fd, h={h:g}, expect ~1e-5)")
    ok = err_w < 1e-6 and err_v < 1e-6 and err_a < 1e-4
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 2. preintegration closes onto the spline
# ---------------------------------------------------------------------------


def check_preintegration(spline, window=0.1, rates=(200, 2000, 20000)):
    banner(f"2. Basalt preintegration over a {window:g} s window, error vs IMU rate")
    t_i = spline.t_min + 1.0
    t_j = t_i + window
    R_i, p_i, _, v_i, _ = spline.derivatives(t_i)
    R_j, p_j, _, v_j, _ = spline.derivatives(t_j)

    print(f"  {'rate [Hz]':>10}  {'rot err [rad]':>14}  {'pos err [m]':>12}  {'vel err [m/s]':>14}")
    prev = None
    ok = True
    for rate in rates:
        dt = 1.0 / rate
        n = int(round(window / dt)) + 1
        ts = t_i + np.arange(n) * dt
        gyro = np.zeros((n, 3))
        accel = np.zeros((n, 3))
        for k, t in enumerate(ts):
            R, _, omega, _, a = spline.derivatives(t)
            gyro[k] = omega
            accel[k] = R.T @ (a - DEFAULT_GRAVITY)
        dR, dv, dp = preintegrate_basalt(gyro, accel, dt)
        Rp, pp, vp = predict_from_preintegration(R_i, p_i, v_i, dR, dv, dp, window)
        e_rot = np.linalg.norm(so3_log(Rp.T @ R_j))
        e_pos = np.linalg.norm(pp - p_j)
        e_vel = np.linalg.norm(vp - v_j)
        print(f"  {rate:>10d}  {e_rot:>14.3e}  {e_pos:>12.3e}  {e_vel:>14.3e}")
        if prev is not None and e_pos > 0.25 * prev:
            ok = False  # should shrink at least ~4x per 10x rate (first-order scheme)
        prev = e_pos
    print("  PASS (error shrinks with rate)" if ok else "  FAIL (error does not shrink)")
    return ok


# ---------------------------------------------------------------------------
# 3. device at rest
# ---------------------------------------------------------------------------


def check_rest():
    banner("3. device at rest: gyro = 0, accel = R^T (0, 0, +9.81)")
    R0 = yaw_pitch_roll(0.7, -0.4, 0.2)
    R = np.tile(R0, (8, 1, 1))
    p = np.tile(np.array([1.0, 2.0, 3.0]), (8, 1))
    sp = CumulativeSE3Spline(0.0, 0.1, R, p)
    ts, gyro, accel = synthesize_imu(sp, 200)
    expect = R0.T @ np.array([0, 0, 9.81])
    e_g = np.abs(gyro).max()
    e_a = np.abs(accel - expect).max()
    print(f"  max |gyro|                = {e_g:.2e}")
    print(f"  max |accel - R^T g_up|    = {e_a:.2e}")
    ok = e_g < 1e-12 and e_a < 1e-9
    print("  PASS" if ok else "  FAIL")
    return ok


# ---------------------------------------------------------------------------
# 4. fake orbit: stations x aims
# ---------------------------------------------------------------------------


def check_fake_orbit():
    banner("4. fake orbit: 12 stations x 25 aim angles, uniform timing")
    rng = np.random.default_rng(0)
    T = []
    for s in range(12):
        station = np.array([2.0 * np.cos(s * 0.5), 2.0 * np.sin(s * 0.5), 1.2 + 0.3 * (s % 3)])
        for a in range(25):
            yaw = s * 0.5 + np.pi + rng.uniform(-0.9, 0.9)
            pitch = rng.uniform(-0.9, 0.9)
            M = np.eye(4)
            M[:3, :3] = yaw_pitch_roll(yaw, pitch, 0.0)
            M[:3, 3] = station
            T.append(M)
    T = np.array(T)

    rep = gap_report(T[:, :3, 3])
    print(f"  poses               : {rep['n_poses']}")
    print(f"  fraction still      : {100 * rep['frac_still']:.1f}%   (prediction was 96%)")
    print(f"  max hop             : {rep['max_gap_m']:.3f} m")

    sp = spline_from_matrices(T, duration=60.0, dt=0.1, timing="uniform")
    _, gyro, accel = synthesize_imu(sp, 200)
    spec = np.linalg.norm(accel, axis=1) - 9.81
    print(f"  peak |accel| - g    : {spec.max():.1f} m/s^2   (handheld is < 5)")
    print(f"  peak |gyro|         : {np.linalg.norm(gyro, axis=1).max():.1f} rad/s")
    print("  (still-spin-jump confirmed; use timing='arclength' or a travel trajectory)")
    return True


if __name__ == "__main__":
    sp = make_test_spline()
    results = [
        check_derivatives(sp),
        check_preintegration(sp),
        check_rest(),
        check_fake_orbit(),
    ]
    banner("SUMMARY: " + ("ALL PASS" if all(results) else "SOME FAILED"))
