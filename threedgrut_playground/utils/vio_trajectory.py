"""VIO-grade deterministic trajectory generator.

Builds a smooth (C3 at every point, including the stationary-to-motion
boundary) rig path around the boards for rendering VIO input bags, and
exports the exact per-frame ground truth. Selected from the GUI's Build
Orbit Trajectory button by setting VIO_TRAJ=1; the orbit generator is
untouched otherwise.

The path: STATIONARY_LEAD_S seconds parked at the initial pose, then a
speed ramp (integrated quintic smoothstep, so position is C3 across the
boundary) into superposed oscillations that excite every axis:
  - distance between a near and a far band (both derived from the board
    cloud and the FIT_HALF_FOV bounds, like the orbit's min_fit_radius),
  - eye azimuth and elevation around the board normal (x/y/z translation),
  - an aim-point wobble whose amplitude is the reference camera's usable
    half field (solved numerically from the device file's projection model)
    minus the boards' angular half extent at the current distance - so the
    boards sweep to the image edges but never leave the reference camera,
  - a gentle roll oscillation about the view axis.
All periods and amplitudes are fixed constants: a longer VIO_MOTION_S adds
cycles (more coverage) without adding speed, per the trackability rule.

Ground truth npz (ate_compare.py's npz layout - keys t_ns, p, q_xyzw):
  - t_ns: START_TIME_NS + i * round(1e9 / fps), identical to the stamps
    the rewritten mcap_convertor writes for frame i (Frames Between = 1).
  - p, q_xyzw: T_world_body - the BODY frame's pose in the playground
    world frame. body = BODY_Q_CAMD * camD-optical (the same NWU body the
    mcap images declare in extrinsic.bodyFrame), world = the polyscope
    scene frame. ate_compare's SE(3) Umeyama alignment absorbs the
    constant world-frame difference to vk_vio's odom frame; the body
    convention matches, so rotation errors are meaningful.
  - extra keys p_camd, q_xyzw_camd: camD optical pose, for debugging.

Environment (existing orbit knobs ORBIT_TARGET / ORBIT_DIST / ORBIT_FLIP
keep working; ARM_REACH is ignored here because clamping breaks C2):
  VIO_TRAJ=1           build this path instead of the orbit
  STATIONARY_LEAD_S    parked lead, default 3.0 s
  VIO_MOTION_S         motion phase, default 40 s (min: one distance cycle)
  VIO_FPS              frame rate, default 20 (also sets PLAYGROUND_FPS so
                       the mcap export stamps at the same rate)
  TRACKABLE_PX         per-frame displacement ceiling, default 50 px
  VIO_TRAJ_NPZ         ground-truth output, default mcap_outputs/vio_truth.npz
"""

import json
import math
import os

import numpy as np

from threedgrut_playground.utils.orbit_trajectory import (
    FIT_HALF_FOV_X_DEG, FIT_HALF_FOV_Y_DEG, UP_AXIS, board_frame, frame_from)

# body_T_camd rotation (x, y, z, w) - keep in sync with
# mcap_convertor.BODY_Q_REF (not imported: that module pulls in capnp).
BODY_Q_CAMD = np.array([0.5, -0.5, 0.5, -0.5])

START_TIME_NS = 1_000_000_000  # mcap_convertor's default start_time_ns

# Motion design constants. Periods in seconds, amplitudes in degrees.
RAMP_S = 4.0            # speed ramp after the stationary lead
T_DIST, T_AZ, T_EL = 20.0, 16.0, 11.0
T_AIMX, T_AIMY, T_ROLL = 9.0, 13.0, 7.0
AZ_EYE_AMP_DEG, EL_EYE_AMP_DEG, ROLL_AMP_DEG = 20.0, 14.0, 5.0
NEAR_BAND_F, FAR_BAND_F = 1.06, 2.1   # x the FIT_HALF_FOV fit distance
AIM_MARGIN_X_DEG, AIM_MARGIN_Y_DEG = 8.0, 6.0
EDGE_PAD_PX = 30.0      # usable-half-field solve stays this far inside


# ---------------------------------------------------------------------------
# Camera models (parameters always read from the device file, never guessed)

def project_kb4(pts, p):
    """(N,3) optical-frame points -> (N,2) pixels + validity, KB4 model."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    t2 = theta * theta
    r_u = theta * (1 + t2 * (p["k1"] + t2 * (p["k2"] + t2 * (p["k3"] + t2 * p["k4"]))))
    safe = np.maximum(rho, 1e-12)
    u = p["fx"] * r_u * x / safe + p["cx"]
    v = p["fy"] * r_u * y / safe + p["cy"]
    return np.stack([u, v], axis=1), theta < 1.6


def project_ds(pts, p):
    """(N,3) optical-frame points -> (N,2) pixels + validity, Double Sphere."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    d1 = np.linalg.norm(pts, axis=1)
    zz = p["xi"] * d1 + z
    d2 = np.sqrt(x * x + y * y + zz * zz)
    denom = p["alpha"] * d2 + (1 - p["alpha"]) * zz
    ok = denom > 1e-6
    denom = np.where(ok, denom, 1.0)
    u = p["fx"] * x / denom + p["cx"]
    v = p["fy"] * y / denom + p["cy"]
    return np.stack([u, v], axis=1), ok


def _project(pts, cam):
    fn = project_ds if cam["model"] == "ds" else project_kb4
    return fn(pts, cam["params"])


def load_rig(calib_path):
    """Device JSON -> per-socket model/params/size/T_camd_cami + ref socket.

    Same field layout the rewritten mcap_convertor validated: kb4 pinhole
    from intrinsicMatrix + distortionCoeff[0:4]; ds from
    distortionCoeff[5:11]; extrinsics cam_i -> reference in CENTIMETRES.
    """
    with open(calib_path) as f:
        data = json.load(f)
    cams, ref = {}, None
    for sock, cam in data["cameraData"]:
        ext = cam["extrinsics"]
        if ext["toCameraSocket"] == -1 or not ext["rotationMatrix"]:
            ref = sock
    if ref is None:
        raise ValueError(f"{calib_path}: no reference camera")
    for sock, cam in data["cameraData"]:
        dc, K = cam["distortionCoeff"], cam["intrinsicMatrix"]
        if cam["cameraType"] == 0:
            model, p = "ds", dict(fx=dc[5], fy=dc[6], cx=dc[7], cy=dc[8],
                                  xi=dc[9], alpha=dc[10])
        else:
            model, p = "kb4", dict(fx=K[0][0], fy=K[1][1], cx=K[0][2],
                                   cy=K[1][2], k1=dc[0], k2=dc[1],
                                   k3=dc[2], k4=dc[3])
        T = np.eye(4)
        ext = cam["extrinsics"]
        if sock != ref:
            T[:3, :3] = np.array(ext["rotationMatrix"])
            T[:3, 3] = [ext["translation"][k] / 100.0 for k in "xyz"]
        cams[sock] = {"model": model, "params": p, "width": cam["width"],
                      "height": cam["height"], "T_camd_cam": T}
    return cams, ref


def usable_half_angle(cam, axis, pad_px=EDGE_PAD_PX):
    """Max angle off the optical axis (deg) that stays pad_px inside the
    image edge, solved by bisection on the camera's own projection model."""
    p, w, h = cam["params"], cam["width"], cam["height"]
    limit = (w - pad_px) if axis == 0 else (h - pad_px)

    def inside(theta):
        d = np.zeros((1, 3))
        d[0, axis] = math.sin(theta)
        d[0, 2] = math.cos(theta)
        px, ok = _project(d, cam)
        return bool(ok[0]) and 0 + pad_px < px[0, axis] < limit
    lo, hi = 0.05, math.radians(120.0)
    if not inside(lo):
        raise ValueError("projection invalid even near the axis")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if inside(mid):
            lo = mid
        else:
            hi = mid
    return math.degrees(lo)


# ---------------------------------------------------------------------------
# Path generation (pure)

class VioConfig:
    def __init__(self):
        env = os.environ.get
        self.lead_s = float(env("STATIONARY_LEAD_S", 3.0))
        self.motion_s = float(env("VIO_MOTION_S", 40.0))
        if self.motion_s < T_DIST:
            print(f"[vio] VIO_MOTION_S {self.motion_s:.0f} s < one distance "
                  f"cycle ({T_DIST:.0f} s): extending (never speeding up)")
            self.motion_s = T_DIST
        self.fps = float(env("VIO_FPS", 20.0))
        self.trackable_px = float(env("TRACKABLE_PX", 50.0))
        self.npz_path = env("VIO_TRAJ_NPZ", "mcap_outputs/vio_truth.npz")
        self.total_s = self.lead_s + self.motion_s
        self.n_frames = int(round(self.total_s * self.fps))
        self.interval_ns = int(round(1e9 / self.fps))


class PathGeometry:
    """Board cloud in the (right, up, fwd) frame + distance bands + budgets."""

    def __init__(self, cloud, center, normal, ref_cam):
        self.cloud = np.asarray(cloud, float)
        self.center = np.asarray(center, float)
        self.fwd = np.asarray(normal, float) / np.linalg.norm(normal)
        self.right, self.up = frame_from(self.fwd)
        rel = self.cloud - self.center
        self.hx = float(np.abs(rel @ self.right).max())
        self.hy = float(np.abs(rel @ self.up).max())
        self.hz = float(np.abs(rel @ self.fwd).max())

        d_fit = self.hz + max(
            self.hx / math.tan(math.radians(FIT_HALF_FOV_X_DEG)),
            self.hy / math.tan(math.radians(FIT_HALF_FOV_Y_DEG)))

        # Usable half fields of the reference camera, minus a margin and
        # minus what the roll oscillation rotates in from the OTHER axis at
        # full corner reach (roll couples the axes: sin(roll) x the other
        # axis's extreme angle lands here). Boards then sweep near the
        # edges yet stay inside.
        ux = usable_half_angle(ref_cam, 0)
        uy = usable_half_angle(ref_cam, 1)
        cross = math.sin(math.radians(ROLL_AMP_DEG))
        self.bud_x = ux - AIM_MARGIN_X_DEG - cross * uy
        self.bud_y = uy - AIM_MARGIN_Y_DEG - cross * ux

        # Near band: FIT-cone distance, but never closer than where the aim
        # budget shrinks below 6 deg (boards would pin the camera to the
        # centre and coverage would die).
        g_min = math.radians(6.0)
        d_bud = self.hz + max(
            self.hx / math.tan(math.radians(self.bud_x) - g_min),
            self.hy / math.tan(math.radians(self.bud_y) - g_min))
        self.d_near = max(NEAR_BAND_F * d_fit, d_bud)
        self.d_far = max(FAR_BAND_F * d_fit, 1.6 * self.d_near)
        dist_env = os.environ.get("ORBIT_DIST")
        if dist_env:
            scale = float(dist_env) / (0.5 * (self.d_near + self.d_far))
            self.d_near *= scale
            self.d_far *= scale
            print(f"[vio] ORBIT_DIST: bands scaled x{scale:.2f}")
        gx = self.bud_x - math.degrees(math.atan(self.hx / (self.d_near - self.hz)))
        gy = self.bud_y - math.degrees(math.atan(self.hy / (self.d_near - self.hz)))
        if gx <= 2.0 or gy <= 2.0:
            raise ValueError(
                f"aim budget collapses at the near band (gx {gx:.1f}, "
                f"gy {gy:.1f} deg): boards too large for these distances - "
                "raise ORBIT_DIST or shrink the boards")


def _tau(t, lead):
    """C3 time warp: 0 through the lead, quintic-smoothstep speed ramp."""
    tp = t - lead
    if tp <= 0.0:
        return 0.0
    if tp >= RAMP_S:
        return 0.5 * RAMP_S + (tp - RAMP_S)
    u = tp / RAMP_S
    return RAMP_S * (u ** 4 * (u * (u - 3.0) + 2.5))


def _look_at_gl(eye, aim, roll_rad):
    """World->camera OpenGL view rotation (rows), with roll about the axis."""
    f = aim - eye
    f = f / np.linalg.norm(f)
    up = UP_AXIS if abs(float(f @ UP_AXIS)) < 0.95 else np.array([1.0, 0, 0])
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    if roll_rad:
        c, sn = math.cos(roll_rad), math.sin(roll_rad)
        s, u = c * s + sn * u, -sn * s + c * u
    return np.stack([s, u, -f])


def pose_at(t, cfg, geom):
    """(eye(3), R_gl(3,3) world->cam rows) at time t. Deterministic, C3."""
    tau = _tau(t, cfg.lead_s)
    w = 2.0 * math.pi
    d_mid = 0.5 * (geom.d_near + geom.d_far)
    d_amp = 0.5 * (geom.d_far - geom.d_near)
    d = d_mid + d_amp * math.sin(w * tau / T_DIST)
    az = math.radians(AZ_EYE_AMP_DEG) * math.sin(w * tau / T_AZ)
    el = math.radians(EL_EYE_AMP_DEG) * math.sin(w * tau / T_EL)
    eye = (geom.center
           + geom.fwd * (d * math.cos(el) * math.cos(az))
           + geom.right * (d * math.cos(el) * math.sin(az))
           + geom.up * (d * math.sin(el)))

    view = geom.center - eye
    dist = float(np.linalg.norm(view))
    gx = math.radians(geom.bud_x) - math.atan(geom.hx / (dist - geom.hz))
    gy = math.radians(geom.bud_y) - math.atan(geom.hy / (dist - geom.hz))
    ax = gx * math.sin(w * tau / T_AIMX)
    ay = gy * math.sin(w * tau / T_AIMY)
    cam_r, cam_u = frame_from(view / dist)
    aim = (geom.center + cam_r * (dist * math.tan(ax))
           + cam_u * (dist * math.tan(ay)))
    roll = math.radians(ROLL_AMP_DEG) * math.sin(w * tau / T_ROLL)
    return eye, _look_at_gl(eye, aim, roll)


F3 = np.diag([1.0, -1.0, -1.0])  # OpenGL camera axes <-> optical (OpenCV)


def sample_path(cfg, geom, times=None):
    """eyes (N,3) and optical camera-to-world rotations R_wc (N,3,3)."""
    if times is None:
        times = np.arange(cfg.n_frames) / cfg.fps
    eyes = np.empty((len(times), 3))
    R_wc = np.empty((len(times), 3, 3))
    for i, t in enumerate(times):
        eye, R_gl = pose_at(float(t), cfg, geom)
        eyes[i] = eye
        R_wc[i] = R_gl.T @ F3   # columns = optical axes in world
    return np.asarray(times, float), eyes, R_wc


def _rot_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
             0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _quat_to_rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def export_npz(path, cfg, eyes, R_wc):
    """Ground truth in ate_compare's npz layout. See the module docstring."""
    t_ns = START_TIME_NS + np.arange(len(eyes), dtype=np.int64) * cfg.interval_ns
    R_cam_body = _quat_to_rot(BODY_Q_CAMD).T   # camd <- body
    q_body = np.empty((len(eyes), 4))
    q_camd = np.empty((len(eyes), 4))
    for i in range(len(eyes)):
        q_camd[i] = _rot_to_quat(R_wc[i])
        q_body[i] = _rot_to_quat(R_wc[i] @ R_cam_body)
    meta = dict(
        frame="p, q_xyzw = T_world_body: body = BODY_Q_CAMD * camD-optical "
              "(the NWU body the rendered images declare in "
              "extrinsic.bodyFrame), world = playground/polyscope scene "
              "frame (constant offset to vk_vio's odom frame; ate_compare "
              "aligns it)",
        stamp=f"t_ns = {START_TIME_NS} + i * {cfg.interval_ns} - identical "
              "to the mcap_convertor stamps at Frames Between = 1",
        fps=cfg.fps, stationary_lead_s=cfg.lead_s, motion_s=cfg.motion_s,
        body_q_camd_xyzw=BODY_Q_CAMD.tolist())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, t_ns=t_ns, p=eyes, q_xyzw=q_body,
             p_camd=eyes, q_xyzw_camd=q_camd, meta=json.dumps(meta))
    return t_ns


# ---------------------------------------------------------------------------
# Verification (no renders: exact projections through the device file)

def _rotvec(Ra, Rb):
    """Rotation vector of Ra^T Rb."""
    R = Ra.T @ Rb
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    ang = math.acos(cos)
    if ang < 1e-12:
        return np.zeros(3)
    return ang / (2 * math.sin(ang)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def verify_path(cfg, geom, rig, ref_sock, times, eyes, R_wc, check_cams=None):
    """All path checks; raises AssertionError on any violation.

    Returns a metrics dict and prints the summary table. check_cams: KB4
    sockets that must reach all four image quadrants and both distance
    bands (default: every KB4 camera except socket 0 - camA points away
    from the boards on this product and is not consumed by the driver).
    """
    n = len(times)
    dt = 1.0 / cfg.fps
    pts_w = geom.cloud
    if check_cams is None:
        check_cams = [s for s, c in rig.items()
                      if c["model"] == "kb4" and s != 0]

    dists = np.linalg.norm(eyes - geom.center, axis=1)
    band_w = geom.d_far - geom.d_near
    near_f = dists <= geom.d_near + 0.15 * band_w
    far_f = dists >= geom.d_far - 0.15 * band_w
    assert near_f.any() and far_f.any(), "distance bands not both visited"

    # per-camera projections over the whole path
    cams = {s: rig[s] for s in set(list(rig)) }
    cov = {}
    for s, cam in cams.items():
        w, h = cam["width"], cam["height"]
        cx, cy = cam["params"]["cx"], cam["params"]["cy"]
        T_cam_camd = np.linalg.inv(cam["T_camd_cam"])
        quad = np.zeros(4, int)
        hist = np.zeros((4, 6), int)
        umin, umax, vmin, vmax = w, 0.0, h, 0.0
        in_near = in_far = 0
        prev_px, prev_ok = None, None
        max_step = 0.0
        inside_all = np.zeros(n, bool)
        for i in range(n):
            Rcw = R_wc[i]
            pts_cam = (pts_w - eyes[i]) @ Rcw          # camD optical
            pts_cam = (pts_cam @ T_cam_camd[:3, :3].T) + T_cam_camd[:3, 3]
            px, ok = _project(pts_cam, cam)
            ok = ok & (pts_cam[:, 2] > 0)
            inside = ok & (px[:, 0] >= 0) & (px[:, 0] < w) \
                        & (px[:, 1] >= 0) & (px[:, 1] < h)
            inside_all[i] = inside.all()
            if inside.any():
                u, v = px[inside, 0], px[inside, 1]
                quad[0] += int(((u < cx) & (v < cy)).sum())
                quad[1] += int(((u >= cx) & (v < cy)).sum())
                quad[2] += int(((u < cx) & (v >= cy)).sum())
                quad[3] += int(((u >= cx) & (v >= cy)).sum())
                bi = np.clip((v / h * 4).astype(int), 0, 3)
                bj = np.clip((u / w * 6).astype(int), 0, 5)
                np.add.at(hist, (bi, bj), 1)
                umin, umax = min(umin, u.min()), max(umax, u.max())
                vmin, vmax = min(vmin, v.min()), max(vmax, v.max())
                if near_f[i]:
                    in_near += int(inside.sum())
                if far_f[i]:
                    in_far += int(inside.sum())
            if prev_px is not None:
                both = inside & prev_ok
                if both.any():
                    step = np.linalg.norm(px[both] - prev_px[both], axis=1).max()
                    max_step = max(max_step, float(step))
            prev_px, prev_ok = px, inside
        cov[s] = dict(quad=quad, hist=hist, span_u=(umin, umax),
                      span_v=(vmin, vmax), near=in_near, far=in_far,
                      max_step_px=max_step, always_inside=bool(inside_all.all()))

    # boards stay in view of the reference camera, every frame
    assert cov[ref_sock]["always_inside"], \
        "board corners leave the reference camera's image"

    for s in check_cams:
        c = cov[s]
        assert (c["quad"] > 0).all(), \
            f"socket {s}: quadrant counts {c['quad'].tolist()} - not all four"
        assert c["near"] > 0 and c["far"] > 0, \
            f"socket {s}: corners missing from a distance band " \
            f"(near {c['near']}, far {c['far']})"

    # kinematics
    v = np.gradient(eyes, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    jerk = np.diff(a, axis=0) / dt
    wvec = np.array([_rotvec(R_wc[i], R_wc[i + 1]) / dt for i in range(n - 1)])
    alpha = np.diff(wvec, axis=0) / dt
    peak = dict(
        v=float(np.linalg.norm(v, axis=1).max()),
        a=float(np.linalg.norm(a, axis=1).max()),
        jerk=float(np.linalg.norm(jerk, axis=1).max()),
        w=float(np.linalg.norm(wvec, axis=1).max()),
        alpha=float(np.linalg.norm(alpha, axis=1).max()))

    # trackability: focal x angular step + translation flow at the nearest
    # board point, per frame (the design formula), plus the measured corner
    # step from the projections above. Both must clear TRACKABLE_PX.
    f_max = max(c["params"]["fx"] for c in cams.values())
    d_min = np.array([np.linalg.norm(pts_w - eyes[i], axis=1).min()
                      for i in range(n)])
    dth = np.linalg.norm(wvec, axis=1) * dt
    dp = np.linalg.norm(np.diff(eyes, axis=0), axis=1)
    px_bound = f_max * dth + f_max * dp / d_min[:-1]
    peak["px_formula"] = float(px_bound.max())
    peak["px_measured"] = max(c["max_step_px"] for c in cov.values())
    assert peak["px_formula"] <= cfg.trackable_px, \
        f"per-frame displacement bound {peak['px_formula']:.1f} px exceeds " \
        f"TRACKABLE_PX {cfg.trackable_px:.0f} - raise VIO_FPS (never the speed)"
    assert peak["px_measured"] <= cfg.trackable_px, \
        f"measured corner step {peak['px_measured']:.1f} px exceeds " \
        f"TRACKABLE_PX {cfg.trackable_px:.0f} - raise VIO_FPS (never the speed)"

    # summary table (also the report the caller prints for the user)
    print(f"[vio] | quantity | value |")
    print(f"[vio] |----|----|")
    print(f"[vio] | duration | {cfg.total_s:.1f} s = {cfg.lead_s:.1f} s "
          f"parked + {cfg.motion_s:.1f} s motion, {n} frames @ "
          f"{cfg.fps:g} fps |")
    print(f"[vio] | distance bands | near {geom.d_near:.2f} m "
          f"({int(near_f.sum())} frames), far {geom.d_far:.2f} m "
          f"({int(far_f.sum())} frames) |")
    print(f"[vio] | peak translation | {peak['v']:.2f} m/s, "
          f"{peak['a']:.2f} m/s^2 |")
    print(f"[vio] | peak rotation | {math.degrees(peak['w']):.1f} deg/s, "
          f"{math.degrees(peak['alpha']):.1f} deg/s^2 |")
    print(f"[vio] | peak per-frame px | formula {peak['px_formula']:.1f}, "
          f"measured {peak['px_measured']:.1f} (ceiling "
          f"{cfg.trackable_px:.0f}) |")
    for s in sorted(cov):
        c, cam = cov[s], cams[s]
        su, sv = c["span_u"], c["span_v"]
        wspan = 100 * (su[1] - su[0]) / cam["width"] if su[1] > su[0] else 0
        print(f"[vio] | cam{'abcd'[s]} ({cam['model']}) | quadrants "
              f"{c['quad'].tolist()}, u {su[0]:.0f}..{su[1]:.0f} "
              f"({wspan:.0f}% of width), near/far corners "
              f"{c['near']}/{c['far']}, max step {c['max_step_px']:.1f} px |")
    for s in sorted(cov):
        print(f"[vio] cam{'abcd'[s]} corner histogram (rows top->bottom):")
        for row in cov[s]["hist"]:
            print("[vio]   " + " ".join(f"{v:6d}" for v in row))
    return dict(peak=peak, coverage=cov, near_frames=int(near_f.sum()),
                far_frames=int(far_f.sum()))


# ---------------------------------------------------------------------------
# GUI glue

def _gather_scene(gui, name_hint):
    """Board vertex cloud + centre + normal from the live scene."""
    from threedgrut_playground.utils.boards import BOARD_NAMES
    objs = gui.primitives.objects
    target = os.environ.get("ORBIT_TARGET")
    if target:
        center, normal, _w, key = board_frame(gui, name_hint)
        prim = objs[key].apply_transform()
        return prim.vertices.detach().cpu().numpy(), center, normal, key
    wanted = [n.lower() for n in BOARD_NAMES if n != "vilota_logo"]
    wanted.append(name_hint.lower())
    keys = [k for k in objs if any(k.lower().startswith(w) for w in wanted)]
    if not keys:
        raise ValueError(f"no board primitives found among {list(objs)}")
    verts, normals = [], []
    for k in keys:
        prim = objs[k].apply_transform()
        verts.append(prim.vertices.detach().cpu().numpy())
        if prim.vertex_normals is not None:
            normals.append(prim.vertex_normals.detach().cpu().numpy().mean(0))
    cloud = np.concatenate(verts, axis=0)
    mn, mx = cloud.min(0), cloud.max(0)
    normal = np.sum(normals, axis=0)
    nn = np.linalg.norm(normal)
    normal = normal / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
    return cloud, (mn + mx) / 2.0, normal, f"{len(keys)} boards {keys}"


def build_vio_trajectory(gui, name_hint="Quad", flip=False):
    """Build the VIO path into the playground trajectory + export truth."""
    cfg = VioConfig()
    cloud, center, normal, label = _gather_scene(gui, name_hint)
    if flip or os.environ.get("ORBIT_FLIP"):
        print("[vio] WARNING ORBIT_FLIP set: camera will be behind the "
              "boards and tags will be mirrored")
        normal = -normal
    if os.environ.get("ARM_REACH"):
        print("[vio] ARM_REACH ignored: clamping would break C2 smoothness")

    nvr = gui.novel_view_renderer
    calib_path = nvr.calibration_fullpath
    rig, ref_sock = load_rig(calib_path)
    geom = PathGeometry(cloud, center, normal, rig[ref_sock])
    print(f"[vio] target {label}, calib {calib_path}")
    print(f"[vio] bands {geom.d_near:.2f}..{geom.d_far:.2f} m, aim budget "
          f"({geom.bud_x:.0f}, {geom.bud_y:.0f}) deg")

    times, eyes, R_wc = sample_path(cfg, geom)
    verify_path(cfg, geom, rig, ref_sock, times, eyes, R_wc)
    t_ns = export_npz(cfg.npz_path, cfg, eyes, R_wc)
    print(f"[vio] ground truth: {cfg.npz_path} ({len(t_ns)} poses, "
          f"T_world_body, stamps matching the mcap export)")

    # the mcap export must stamp at the same rate the path was sampled at
    os.environ["PLAYGROUND_FPS"] = str(cfg.fps)
    print(f"[vio] PLAYGROUND_FPS={cfg.fps:g} set for the mcap export")

    nvr.create_new_trajectory()
    origin = nvr.get_origin_camera_index()
    poses = None
    # one tail pose past the end: the renderer emits len(poses)-1 frames
    tail_t = cfg.n_frames / cfg.fps
    for i in range(cfg.n_frames + 1):
        t = tail_t if i == cfg.n_frames else float(times[i])
        eye, R_gl = pose_at(t, cfg, geom)
        view = np.eye(4)
        view[:3, :3] = R_gl
        view[:3, 3] = -R_gl @ eye
        poses = nvr.add_pose_to_trajectory(view, origin)
    gui.orbit_first_pose = {c: 0 for c in range(4)}
    print(f"[vio] {cfg.n_frames} frames ({cfg.n_frames + 1} poses incl. the "
          f"unrendered tail) into the trajectory")
    return poses
