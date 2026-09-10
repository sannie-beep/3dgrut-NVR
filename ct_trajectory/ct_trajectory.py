"""Continuous-time trajectory for synthetic IMU generation.

Cumulative cubic B-spline on SO(3) x R^3 with uniform knots, after
Lovegrove et al. (Spline Fusion), Sommer et al. (CVPR 2020) and
Hug et al. (Hyperion, Sec. 3.2, Eqs. 4-6).

Conventions
-----------
T_wb          : body-to-world. p_w = R_wb @ p_b + t_wb.
Control point : one pose per knot, knots spaced `dt` seconds apart.
Valid range   : a cubic segment needs four control points, so the
                spline is defined on [t0 + dt, t0 + (N-2)*dt).
Gravity       : world frame, default (0, 0, -9.81). Change it if your
                scene is not z-up.

Only numpy is required.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------

_EPS = 1e-10


def hat(w: np.ndarray) -> np.ndarray:
    """3-vector -> skew-symmetric matrix."""
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation vector -> rotation matrix (Rodrigues)."""
    th = np.linalg.norm(w)
    K = hat(w)
    if th < _EPS:
        return np.eye(3) + K
    s, c = np.sin(th), np.cos(th)
    return np.eye(3) + (s / th) * K + ((1.0 - c) / (th * th)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector."""
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(tr)
    if th < _EPS:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-6:
        # Near pi: use the symmetric part.
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        # Fix signs from the off-diagonals.
        if axis[0] > 0:
            axis[1] = np.copysign(axis[1], A[0, 1])
            axis[2] = np.copysign(axis[2], A[0, 2])
        elif axis[1] > 0:
            axis[2] = np.copysign(axis[2], A[1, 2])
        return th * axis / np.linalg.norm(axis)
    return (th / (2.0 * np.sin(th))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


# ---------------------------------------------------------------------------
# Cumulative cubic basis (Lovegrove / Patron-Perez / Sibley)
# ---------------------------------------------------------------------------

# lambda(u) = C @ [1, u, u^2, u^3]. Row 0 is the constant 1.
_C = (1.0 / 6.0) * np.array(
    [[6, 0, 0, 0], [5, 3, -3, 1], [1, 3, 3, -2], [0, 0, 0, 1]], dtype=float
)


def _basis(u: float):
    """Return lambda, dlambda/du, d2lambda/du2 for u in [0, 1)."""
    U = np.array([1.0, u, u * u, u * u * u])
    dU = np.array([0.0, 1.0, 2.0 * u, 3.0 * u * u])
    ddU = np.array([0.0, 0.0, 2.0, 6.0 * u])
    return _C @ U, _C @ dU, _C @ ddU


# ---------------------------------------------------------------------------
# The spline
# ---------------------------------------------------------------------------


class CumulativeSE3Spline:
    """Cubic B-spline with split SO(3) / R^3 interpolation."""

    def __init__(self, t0: float, dt: float, R_ctrl: np.ndarray, p_ctrl: np.ndarray):
        R_ctrl = np.asarray(R_ctrl, dtype=float)
        p_ctrl = np.asarray(p_ctrl, dtype=float)
        if R_ctrl.shape[1:] != (3, 3) or p_ctrl.shape[1:] != (3,):
            raise ValueError("R_ctrl must be (N,3,3), p_ctrl must be (N,3)")
        if len(R_ctrl) != len(p_ctrl) or len(R_ctrl) < 4:
            raise ValueError("need at least 4 control points")
        self.t0 = float(t0)
        self.dt = float(dt)
        self.R = R_ctrl
        self.p = p_ctrl
        self.N = len(R_ctrl)
        # Body-frame rotation increments d_k = log(R_{k-1}^T R_k), k = 1..N-1.
        self.d = np.zeros((self.N, 3))
        self.dp = np.zeros((self.N, 3))
        for k in range(1, self.N):
            self.d[k] = so3_log(self.R[k - 1].T @ self.R[k])
            self.dp[k] = self.p[k] - self.p[k - 1]

    # -- time range ---------------------------------------------------------

    @property
    def t_min(self) -> float:
        return self.t0 + self.dt

    @property
    def t_max(self) -> float:
        return self.t0 + (self.N - 2) * self.dt

    def _segment(self, t: float):
        tol = 1e-9 * self.dt
        if not (self.t_min - tol <= t < self.t_max + tol):
            raise ValueError(f"t={t} outside valid range [{self.t_min}, {self.t_max})")
        s = (t - self.t0) / self.dt
        i = int(np.floor(s))
        i = min(max(i, 1), self.N - 3)
        u = s - i
        return i, u

    # -- evaluation -----------------------------------------------------------

    def pose(self, t: float):
        """Return (R_wb, p_wb) at time t."""
        i, u = self._segment(t)
        lam, _, _ = _basis(u)
        A1 = so3_exp(lam[1] * self.d[i])
        A2 = so3_exp(lam[2] * self.d[i + 1])
        A3 = so3_exp(lam[3] * self.d[i + 2])
        R = self.R[i - 1] @ A1 @ A2 @ A3
        p = self.p[i - 1] + lam[1] * self.dp[i] + lam[2] * self.dp[i + 1] + lam[3] * self.dp[i + 2]
        return R, p

    def derivatives(self, t: float):
        """Return (R_wb, p_wb, omega_body, v_world, a_world) at time t.

        omega_body is the angular velocity in the body frame, i.e. the
        gyroscope reading. v_world and a_world are the first and second
        time derivatives of p_wb.
        """
        i, u = self._segment(t)
        lam, dlam, ddlam = _basis(u)
        dlam = dlam / self.dt
        ddlam = ddlam / (self.dt * self.dt)

        d1, d2, d3 = self.d[i], self.d[i + 1], self.d[i + 2]
        A1 = so3_exp(lam[1] * d1)
        A2 = so3_exp(lam[2] * d2)
        A3 = so3_exp(lam[3] * d3)
        R = self.R[i - 1] @ A1 @ A2 @ A3

        # R^T dR/dt = [ (A2 A3)^T dl1 d1 + A3^T dl2 d2 + dl3 d3 ]_x
        omega = (A2 @ A3).T @ (dlam[1] * d1) + A3.T @ (dlam[2] * d2) + dlam[3] * d3

        p = self.p[i - 1] + lam[1] * self.dp[i] + lam[2] * self.dp[i + 1] + lam[3] * self.dp[i + 2]
        v = dlam[1] * self.dp[i] + dlam[2] * self.dp[i + 1] + dlam[3] * self.dp[i + 2]
        a = ddlam[1] * self.dp[i] + ddlam[2] * self.dp[i + 1] + ddlam[3] * self.dp[i + 2]
        return R, p, omega, v, a


# ---------------------------------------------------------------------------
# Building a spline from poses
# ---------------------------------------------------------------------------


def split_matrices(T: np.ndarray):
    """(N,4,4) body-to-world -> (R (N,3,3), p (N,3))."""
    T = np.asarray(T, dtype=float)
    return T[:, :3, :3].copy(), T[:, :3, 3].copy()


def uniform_times(n: int, duration: float) -> np.ndarray:
    """Equal time per pose. Simple, and wrong for uneven spacing."""
    return np.linspace(0.0, duration, n)


def arclength_times(p: np.ndarray, duration: float, min_step: float = 1e-6) -> np.ndarray:
    """Time proportional to distance travelled, so speed is roughly constant.

    Repeated positions get `min_step` of time so knots stay strictly
    increasing. That does not make the device move. It only stops the
    resampler from dividing by zero.
    """
    p = np.asarray(p, dtype=float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    seg = np.maximum(seg, min_step)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s / s[-1] * duration


def resample_uniform(times: np.ndarray, R: np.ndarray, p: np.ndarray, dt: float):
    """Resample irregularly timed poses onto a uniform knot grid.

    Linear in position, geodesic (log-lerp) in rotation. Returns
    (t0, R_ctrl, p_ctrl) ready for CumulativeSE3Spline.
    """
    times = np.asarray(times, dtype=float)
    grid = np.arange(times[0], times[-1] + 0.5 * dt, dt)
    grid = grid[grid <= times[-1]]
    R_out = np.zeros((len(grid), 3, 3))
    p_out = np.zeros((len(grid), 3))
    j = 0
    for k, t in enumerate(grid):
        while j < len(times) - 2 and times[j + 1] <= t:
            j += 1
        span = times[j + 1] - times[j]
        w = 0.0 if span <= 0 else np.clip((t - times[j]) / span, 0.0, 1.0)
        p_out[k] = (1.0 - w) * p[j] + w * p[j + 1]
        R_out[k] = R[j] @ so3_exp(w * so3_log(R[j].T @ R[j + 1]))
    return grid[0], R_out, p_out


def spline_from_matrices(T: np.ndarray, duration: float, dt: float = 0.1, timing: str = "uniform"):
    """One-call builder. `timing` is 'uniform' or 'arclength'."""
    R, p = split_matrices(T)
    if timing == "uniform":
        times = uniform_times(len(p), duration)
    elif timing == "arclength":
        times = arclength_times(p, duration)
    else:
        raise ValueError("timing must be 'uniform' or 'arclength'")
    t0, Rc, pc = resample_uniform(times, R, p, dt)
    return CumulativeSE3Spline(t0, dt, Rc, pc)


# ---------------------------------------------------------------------------
# Sampling: camera poses and IMU
# ---------------------------------------------------------------------------


def sample_poses(spline: CumulativeSE3Spline, rate_hz: float):
    """Camera poses at a fixed rate. Returns (times, T (M,4,4))."""
    ts = _time_grid(spline, rate_hz)
    T = np.tile(np.eye(4), (len(ts), 1, 1))
    for k, t in enumerate(ts):
        R, p = spline.pose(t)
        T[k, :3, :3] = R
        T[k, :3, 3] = p
    return ts, T


DEFAULT_GRAVITY = np.array([0.0, 0.0, -9.81])


def _time_grid(spline: CumulativeSE3Spline, rate_hz: float) -> np.ndarray:
    """Sample times strictly inside the valid range, immune to arange round-off."""
    n = int(np.floor((spline.t_max - spline.t_min) * rate_hz - 1e-9)) + 1
    return spline.t_min + np.arange(n) / rate_hz


def synthesize_imu(spline: CumulativeSE3Spline, rate_hz: float, gravity: np.ndarray = DEFAULT_GRAVITY):
    """Noiseless IMU stream. Returns (times, gyro (M,3), accel (M,3)).

    gyro  = angular velocity in the body frame            [rad/s]
    accel = R_wb^T (a_world - gravity), the specific force [m/s^2]
    A device at rest reports accel = R^T (0,0,+9.81).
    """
    ts = _time_grid(spline, rate_hz)
    gyro = np.zeros((len(ts), 3))
    accel = np.zeros((len(ts), 3))
    for k, t in enumerate(ts):
        R, _, omega, _, a = spline.derivatives(t)
        gyro[k] = omega
        accel[k] = R.T @ (a - gravity)
    return ts, gyro, accel


# ---------------------------------------------------------------------------
# Basalt preintegration (Usenko et al. 2020, Eqs. 10-12, 18-20)
# ---------------------------------------------------------------------------


def preintegrate_basalt(gyro: np.ndarray, accel: np.ndarray, dt: float):
    """Eqs. 10-12. Measurement k is applied over the step that ends at k.

    Returns (dR, dv, dp) in the frame of the first sample.
    """
    dR = np.eye(3)
    dv = np.zeros(3)
    dp = np.zeros(3)
    for k in range(1, len(gyro)):
        dp = dp + dv * dt                       # Eq. 12, uses old dv
        dv = dv + dR @ accel[k] * dt            # Eq. 11, uses old dR
        dR = dR @ so3_exp(gyro[k] * dt)         # Eq. 10
    return dR, dv, dp


def predict_from_preintegration(R_i, p_i, v_i, dR, dv, dp, T, gravity=DEFAULT_GRAVITY):
    """Invert Eqs. 18-20 with zero residual to predict the end state."""
    R_j = R_i @ dR
    v_j = v_i + gravity * T + R_i @ dv
    p_j = p_i + v_i * T + 0.5 * gravity * T * T + R_i @ dp
    return R_j, p_j, v_j


# ---------------------------------------------------------------------------
# Diagnostic: how still is the trajectory?
# ---------------------------------------------------------------------------


def gap_report(p: np.ndarray, still_thresh: float = 1e-3) -> dict:
    """Consecutive position gaps. Tells you if the poses are a journey or a list."""
    p = np.asarray(p, dtype=float)
    gaps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return {
        "n_poses": int(len(p)),
        "frac_still": float(np.mean(gaps < still_thresh)),
        "max_gap_m": float(gaps.max()),
        "median_gap_m": float(np.median(gaps)),
        "total_path_m": float(gaps.sum()),
    }
