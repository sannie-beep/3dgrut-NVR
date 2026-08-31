"""Build an orbit trajectory that serves every camera on the rig.

Poses are grouped by camera. Each group is built in world space around the
board, then stored through add_pose_to_trajectory with that camera's index,
which converts the view into the rig frame. The named camera therefore sees
the board exactly as aimed, and every camera gets direct coverage instead of
whatever falls out of serving CamD alone.

Aim grids are per camera type. The KB4 cameras are 1280x800 with about a 58
degree half field, so wide aims would only point them at nothing. The Double
Sphere camera keeps the wide rectangular grid.
"""
import math
import numpy as np
import polyscope as ps

# Camera indices on the rig and which aim grid each uses.
KB4_CAMS = [0, 1, 2]
DS_CAMS = [3]

# Viewpoints, shared by all cameras.
ELEVATIONS = [-20.0, 20.0]
AZIMUTH_STEPS = 3
AZIMUTH_SPAN = 90.0
DISTANCE_FACTORS = [0.5, 0.9]

# Aim grids, degrees off the optical axis. Yaw is horizontal.
AIM_KB4 = ([-45.0, -22.0, 0.0, 22.0, 45.0],
           [-28.0, 0.0, 28.0])
AIM_DS = ([-78.0, -58.0, -38.0, -19.0, 0.0, 19.0, 38.0, 58.0, 78.0],
          [-49.0, -33.0, -16.0, 0.0, 16.0, 33.0, 49.0])
MAX_AIM_ANGLE = 95.0

# The scene up axis, from the Up setting in the Render widget (neg_y_up).
UP_AXIS = np.array([0.0, 1.0, 0.0])


def board_frame(gui, name_hint="Quad"):
    """Return the center, normal, width and key of the target primitive."""
    objs = gui.primitives.objects
    key = next((k for k in objs if name_hint.lower() in k.lower()), None)
    if key is None:
        raise ValueError(f"No primitive matching '{name_hint}'. Have: {list(objs)}")

    prim = objs[key].apply_transform()
    verts = prim.vertices.detach().cpu().numpy()
    center = verts.mean(axis=0)

    extent = verts.max(axis=0) - verts.min(axis=0)
    width = float(np.sort(extent)[-1])

    if prim.vertex_normals is not None:
        normal = prim.vertex_normals.detach().cpu().numpy().mean(axis=0)
    else:
        tris = prim.triangles.detach().cpu().numpy()
        a, b, c = verts[tris[0][0]], verts[tris[0][1]], verts[tris[0][2]]
        normal = np.cross(b - a, c - a)
    n = np.linalg.norm(normal)
    normal = normal / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    return center, normal, width, key


def frame_from(forward):
    """Return a right and up axis perpendicular to forward."""
    world_up = UP_AXIS
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up


def aim_angles(yaws, pitches):
    """Rectangular aim grid in serpentine order."""
    pairs = []
    flip = False
    for pitch in pitches:
        row = list(yaws)
        if flip:
            row.reverse()
        for yaw in row:
            if math.hypot(yaw, pitch) <= MAX_AIM_ANGLE:
                pairs.append((yaw, pitch))
        flip = not flip
    return pairs


def eye_positions(center, normal, width):
    """Camera positions on a serpentine arc in front of the board."""
    fwd = normal
    right, up = frame_from(fwd)
    eyes = []
    flip = False
    for factor in DISTANCE_FACTORS:
        r = width * factor
        for el in ELEVATIONS:
            el_rad = math.radians(el)
            steps = list(range(AZIMUTH_STEPS))
            if flip:
                steps.reverse()
            for k in steps:
                frac = k / max(AZIMUTH_STEPS - 1, 1)
                az = math.radians(-AZIMUTH_SPAN / 2.0 + AZIMUTH_SPAN * frac)
                offset = (
                    fwd * (r * math.cos(el_rad) * math.cos(az))
                    + right * (r * math.cos(el_rad) * math.sin(az))
                    + up * (r * math.sin(el_rad))
                )
                eyes.append(center + offset)
            flip = not flip
    return eyes


def make_pose(center, eye, yaw, pitch):
    """Return an aim point, or None when look_at would be degenerate."""
    view = center - eye
    dist = float(np.linalg.norm(view))
    if dist < 1e-6:
        return None
    view = view / dist
    cam_right, cam_up = frame_from(view)
    aim = (
        center
        + cam_right * (dist * math.tan(math.radians(yaw)))
        + cam_up * (dist * math.tan(math.radians(pitch)))
    )
    look = aim - eye
    n = np.linalg.norm(look)
    if n < 1e-6:
        return None
    if abs(float(np.dot(look / n, UP_AXIS))) > 0.97:
        return None
    return aim


def build_orbit_trajectory(gui, cam_index=None, name_hint="Quad", flip=False):
    """Fill the trajectory, serving every camera on the rig in turn."""
    center, normal, width, key = board_frame(gui, name_hint)
    if flip:
        normal = -normal

    nvr = gui.novel_view_renderer
    nvr.create_new_trajectory()

    eyes = eye_positions(center, normal, width)
    plan = [(c, aim_angles(*AIM_KB4)) for c in KB4_CAMS]
    plan += [(c, aim_angles(*AIM_DS)) for c in DS_CAMS]

    poses = None
    per_cam = {}
    for cam, angles in plan:
        n_added = 0
        flip_seq = False
        for eye in eyes:
            seq = list(reversed(angles)) if flip_seq else angles
            flip_seq = not flip_seq
            for yaw, pitch in seq:
                aim = make_pose(center, eye, yaw, pitch)
                if aim is None:
                    continue
                ps.look_at(eye, aim)
                view_mat = ps.get_view_camera_parameters().get_view_mat()
                poses = nvr.add_pose_to_trajectory(view_mat, cam)
                n_added += 1
        per_cam[cam] = n_added

    total = sum(per_cam.values())
    print(f"[orbit] target '{key}' center={center} width={width:.3f}m")
    print(f"[orbit] {len(eyes)} viewpoints, per-camera poses: {per_cam}")
    print(f"[orbit] added {total} poses")
    return poses
