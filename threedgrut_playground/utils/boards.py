"""Board textures for the rig's six AprilGrid configurations, plus a logo.

The rig declares six grids that share one tag pool. They differ by layout, not
by tag id, and the solver tells them apart by a hash of the layout:

    3x1   ids 0 3 6                 increment 3
    2x2   ids 0 2 4 6               increment 2
    4x7   ids 0..27                 base order
    4x7   ids 0 14 1 15 ... 13 27   offset 14
    4x7   ids 0 15 1 16 ... 13 28   offset 15
    4x7   ids 0 16 1 17 ... 13 29   offset 16

This file does not change create_aprilgrid.py. It reuses create_grid for the
black squares and add_padding for the border, and runs its own tag pass so the
tag order is an argument instead of a hardcoded list.
"""

import os

import torch

from threedgrut_playground.utils.create_aprilgrid import (
    add_padding, create_grid, create_tag)

TAG_ORDERS = {
    "base":  list(range(28)),
    "off14": [v for i in range(14) for v in (i, i + 14)],
    "off15": [v for i in range(14) for v in (i, i + 15)],
    "off16": [v for i in range(14) for v in (i, i + 16)],
}

BOARD_SPECS = [
    ("grid_3x1",   3, 1, [0, 3, 6]),
    ("grid_2x2",   2, 2, [0, 2, 4, 6]),
    ("aprilgrid",  4, 7, TAG_ORDERS["base"]),   # the 4x7 base board
    ("grid_off14", 4, 7, TAG_ORDERS["off14"]),
    ("grid_off15", 4, 7, TAG_ORDERS["off15"]),
    ("grid_off16", 4, 7, TAG_ORDERS["off16"]),
    ("grid_3x3",   3, 3, list(range(9))),
]


def row_major_to_texture_order(ids, rows, cols):
    """Reorder ids for the texture, which walks the top row first.

    The solver lists tag ids bottom row first, left to right. The texture walks
    the other way. This reproduces create_aprilgrid.py's hardcoded list
    [21, 22, ..., 6] exactly from range(28).
    """
    grid = [ids[r * cols:(r + 1) * cols] for r in range(rows)]
    return [v for row in reversed(grid) for v in row]


def _tag_pass(square, gap, rows, cols, texture, offset, tag_order, device):
    """Draw one tag per cell, in the given order. Mirrors create_grid's layout."""
    idx = 0
    for row in range(rows):
        for col in range(cols):
            start_row = row * int(square + gap + 2 * offset) + offset
            start_col = col * int(square + gap + 2 * offset) + offset
            create_tag(start_row, start_row + square,
                       start_col, start_col + square,
                       square, texture, tagID=tag_order[idx], device=device)
            idx += 1
    return texture


def make_texture(gap_ratio, square, device, rows, cols, tag_order):
    if len(tag_order) < rows * cols:
        raise ValueError(f"tag_order holds {len(tag_order)} ids, "
                         f"a {rows}x{cols} board needs {rows * cols}")
    board_gap = square * gap_ratio
    width = int(square * cols + board_gap * (cols - 1))
    height = int(square * rows + board_gap * (rows - 1))
    square, gap = int(square), int(board_gap)

    tex = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device,
                       dtype=torch.float32).repeat(height, width, 1)
    tex = create_grid(square, gap, 0.0, rows, cols, tex)
    tex = _tag_pass(int(square * 0.5), gap, rows, cols, tex,
                    int(square / 4), tag_order, device)
    return add_padding(width, height, gap, tex, device=device)


def build_materials(device, PBRMaterial, logo_path=None, square=800):
    """One material per board plus the logo. Ids run from 0 with no gaps.

    The gap matters: engine.py sorts materials by material_id and passes the
    list positionally, so a hole makes every later index name the wrong
    material. The original file had one material at id 1 and nothing at id 0,
    which only worked because material_assignments gets clamped to 0.
    """
    def mat(mid, tex):
        return PBRMaterial(
            material_id=mid,
            diffuse_map=tex.contiguous(),
            diffuse_factor=torch.ones(4, device=device, dtype=torch.float32),
            emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
            metallic_factor=0.0,
            roughness_factor=0.0,
            transmission_factor=0.0,
            ior=1.0,
        )

    materials = {}
    for mid, (name, rows, cols, ids) in enumerate(BOARD_SPECS):
        order = row_major_to_texture_order(ids, rows, cols)
        materials[name] = mat(mid, make_texture(0.3, square, device,
                                                rows, cols, order))

    # Do NOT alias. ps_gui.py:1377 maps material_id to dict POSITION, so a
    # second dict entry pointing at the same object shifts every later index
    # and the material dropdown selects the wrong thing.

    if logo_path and os.path.exists(logo_path):
        from PIL import Image
        import numpy as np
        # Flatten alpha onto white. A logo PNG has a transparent background,
        # and this material does not set alpha_mode, so the quad renders as
        # nothing.
        src = Image.open(logo_path).convert("RGBA")
        bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
        img = np.asarray(Image.alpha_composite(bg, src), dtype=np.float32) / 255.0
        materials["vilota_logo"] = mat(
            len(BOARD_SPECS),
            torch.from_numpy(img).to(device=device, dtype=torch.float32))
    else:
        print(f"[playground] no logo at {logo_path}, skipping the logo material")

    return materials


BOARD_NAMES = [s[0] for s in BOARD_SPECS] + ["vilota_logo"]


# ---------------------------------------------------------------------------
# Scene spawning

DEFAULT_SCENE = ["aprilgrid", "grid_off14", "grid_off15"]


def _axis(spec, default):
    spec = (spec or default).strip().lower()
    return {"x": 0, "y": 1, "z": 2}[spec.lstrip("+-")], (-1.0 if spec.startswith("-") else 1.0)


def board_half_extents(name, tag_cm):
    for spec_name, rows, cols, _ids in BOARD_SPECS:
        if spec_name == name:
            return ((cols + 0.3 * (cols - 1) + 0.6) * tag_cm * 0.01 / 2.0,
                    (rows + 0.3 * (rows - 1) + 0.6) * tag_cm * 0.01 / 2.0)
    return None


def spawn_scene(engine, primitive_type, device):
    """Place boards at different heights AND different distances.

    Positions are angles off the optical axis, not metres, so the layout holds
    at any tag size. The near distance defaults to about 0.2 * tag_cm metres,
    which is where a tag still spans roughly 20 px on a 393 px focal camera.
    Board height and distance then scale together, so the angular size of a
    board stays near 7.8 degrees whatever size you pick.

    Environment:
        PLAYGROUND_BOARDS   1, a comma separated list of names, or 'all'
        BOARD_TAG_CM        tag pitch. Default 5.173, the rig's real size
        BOARD_DISTANCE      near distance in metres. Default 0.2 * BOARD_TAG_CM
        BOARD_DEPTH_SPAN    far board sits at distance * (1 + this). Default 0.6
        BOARD_ANGLES        degrees off axis, one per board. Default -22,0,22
        BOARD_UP_AXIS       default y
        BOARD_FWD_AXIS      default -z
    """
    import math
    import os

    scene_file = os.environ.get("BOARD_SCENE")
    if scene_file:
        return spawn_from_file(engine, primitive_type, device, scene_file)

    want = os.environ.get("PLAYGROUND_BOARDS", "").strip()
    if want.lower() == "all":
        names = list(BOARD_NAMES)
    elif want and want != "1":
        names = [n.strip() for n in want.split(",") if n.strip()]
    else:
        names = list(DEFAULT_SCENE)

    tag_cm = float(os.environ.get("BOARD_TAG_CM", 5.173))
    near = float(os.environ.get("BOARD_DISTANCE", 0.20 * tag_cm))
    span = float(os.environ.get("BOARD_DEPTH_SPAN", 0.6))
    up_i, up_s = _axis(os.environ.get("BOARD_UP_AXIS"), "y")
    fw_i, fw_s = _axis(os.environ.get("BOARD_FWD_AXIS"), "-z")

    n = len(names)
    raw = os.environ.get("BOARD_ANGLES")
    if raw:
        angles = [float(a) for a in raw.split(",")]
    elif n == 1:
        angles = [0.0]
    else:
        angles = [-22.0 + 44.0 * i / (n - 1) for i in range(n)]
    angles = (angles + [0.0] * n)[:n]

    print(f"[playground] {n} board(s), tag {tag_cm:.3f} cm, "
          f"near {near:.2f} m, far {near * (1 + span):.2f} m")

    for i, (name, ang) in enumerate(zip(names, angles)):
        if name not in engine.primitives.registered_materials:
            print(f"[playground]   skip {name}, no such material")
            continue
        # Nearest board at the bottom, farthest at the top.
        frac = 0.0 if n == 1 else i / (n - 1)
        dist = near * (1.0 + span * frac)

        before = set(engine.primitives.objects)
        engine.primitives.add_primitive(
            geometry_type="Quad", primitive_type=primitive_type,
            device=device, material_name=name)
        added = set(engine.primitives.objects) - before
        if not added:
            print(f"[playground]   {name} did not spawn")
            continue
        obj = engine.primitives.objects[added.pop()]

        pos = [0.0, 0.0, 0.0]
        pos[fw_i] = fw_s * dist
        pos[up_i] = up_s * dist * math.tan(math.radians(ang))
        obj.transform.tx, obj.transform.ty, obj.transform.tz = pos

        size = board_half_extents(name, tag_cm)
        if size is not None:
            obj.transform.sx, obj.transform.sy = size
        print(f"[playground]   {name:<12} d {dist:5.2f} m  {ang:+6.1f} deg  "
              f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
              + (f"  sx {size[0]:.4f} sy {size[1]:.4f}" if size else "  (size by hand)"))


def spawn_from_file(engine, primitive_type, device, path):
    """Place boards from a JSON scene file. See office_scene.json.

    Each entry: material (required), pos [x,y,z] (required), and either
    tag_cm for a board or sx/sy for anything else. rot [rx,ry,rz] degrees
    is optional and defaults to flat.
    """
    import json
    spec = json.load(open(path))
    boards = spec.get("boards", [])
    print(f"[playground] scene file {path}, {len(boards)} board(s)")
    for entry in boards:
        name = entry["material"]
        if name not in engine.primitives.registered_materials:
            print(f"[playground]   skip {name}, no such material")
            continue
        before = set(engine.primitives.objects)
        engine.primitives.add_primitive(
            geometry_type="Quad", primitive_type=primitive_type,
            device=device, material_name=name)
        added = set(engine.primitives.objects) - before
        if not added:
            print(f"[playground]   {name} did not spawn")
            continue
        obj = engine.primitives.objects[added.pop()]

        pos = entry["pos"]
        obj.transform.tx, obj.transform.ty, obj.transform.tz = pos
        rot = entry.get("rot", [0.0, 0.0, 0.0])
        obj.transform.rx, obj.transform.ry, obj.transform.rz = rot

        if "tag_cm" in entry:
            size = board_half_extents(name, float(entry["tag_cm"]))
        elif "sx" in entry:
            size = (float(entry["sx"]), float(entry["sy"]))
        else:
            size = None
        if size is not None:
            obj.transform.sx, obj.transform.sy = size
        print(f"[playground]   {name:<12} pos ({pos[0]:+.2f}, {pos[1]:+.2f}, "
              f"{pos[2]:+.2f}) rot ({rot[0]:+.1f}, {rot[1]:+.1f}, {rot[2]:+.1f})"
              + (f" sx {size[0]:.4f} sy {size[1]:.4f}" if size else ""))
