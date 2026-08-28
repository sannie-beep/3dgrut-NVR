# vilota-ref playground

Render AprilGrid boards through a real device calibration, detect them, and
check vk_calibrate recovers the same camera.

## Rules
- NEVER run a render or launch playground.py. The GPU is the user's.
- NEVER run vk_calibrate.
- Launch line, for reference only. Without the two NV variables polyscope
  segfaults on Mesa:
    PLAYGROUND_BOARDS=1 __NV_PRIME_RENDER_OFFLOAD=1 \
      __GLX_VENDOR_LIBRARY_NAME=nvidia python playground.py \
      --gs_object ply_files/room.ply
- One change per commit, reason in the message.
- After every edit run: python3 -m py_compile <each file touched>
- Before claiming an edit applied, grep the file and show the new text.
  Do not trust a patch script's own success message. This has already
  produced silent no-ops twice.
- Never guess an intrinsic value. Read it from the device file.

## Calibration files, ./calibration_files/
    DP180IP-30020104.json   30.02.0104          4 cams  <- the real Aug 17 unit
    vk180.json              DP180IP-2404-0004   4 cams  <- every render so far
    dp180_1.json            DP180-2501-0106     3 cams  <- only 3
    dp180_2.json            DP180-2501-0109     4 cams
    vkl.json                no deviceName       4 cams
Filename and serial do not agree. Truth is the device file for the unit that
made the data.

## The rig
Six AprilGrid layouts sharing one tag pool, told apart by a hash of the
layout, not by tag id:
    3x1  ids 0 3 6            2x2  ids 0 2 4 6
    4x7  ids 0..27            4x7  interleaved at offsets 14, 15, 16
Real tag pitch 5.173 cm, spacing 0.3, family tag16h5.

## Recent work, all in threedgrut_playground/
- utils/boards.py is NEW. One material per board plus a logo, ids 0..6.
  aprilgrid MUST stay first in BOARD_SPECS: a quad with no material_name
  gets id 0, and that is the original single-board behaviour.
- Materials must have DENSE ids from 0. engine.py sorts by material_id and
  indexes positionally, and ps_gui.py:1377 maps id to dict POSITION. A gap
  or a duplicate entry makes the material dropdown select the wrong thing.
- ps_gui.py: Translate/Rotate/Scale sliders restored (they existed but sat
  outside the function). Square size button fixed and generalised. Reset now
  keeps translation and rotation. Calibration file dropdown added.
- mesh_io.py zeroes material_assignments for every procedural mesh, so
  add_primitive takes material_name to override it.

## Known bugs, still open
- calculate_sx_and_sy had rows/cols swapped AND the call site swapped them
  back. Fixed, but check nothing else calls it.
- move_rig_to_pose tests `if not cam_index`, so camera 0 silently becomes
  the origin camera.
- The MCAP writer sets encoding, step and frameId wrongly. fix_mcap_labels.py
  is the workaround.
- ps_gui.py writes every export to mcap_outputs/long_final_path.mcap and
  overwrites it each render.
- A 3x3 grid with ids 0..8 CANNOT be added to offline_tags_all.json: its
  adjacencies (0-1, 1-2, 0-3, 3-6 ...) duplicate the 4x7 base and 3x1 grids,
  so vk_camera_driver aborts at startup ("Duplicate adj ids found for multi
  grid config") and the recorder writes an empty bag. boards.py grid_3x3
  still renders that layout; a conflict-free order is [0,3,1,2,4,7,5,8,6].

## Do not break
Default launch, with no PLAYGROUND_BOARDS, must give exactly one 4x7 board
and the working build-trajectory flow. That is the user's known-good path.
