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

## Orbit generator knobs (utils/orbit_trajectory.py, env vars, uncommitted)
- ORBIT_TARGET: substring of the primitive to orbit (fallback grid_off14, then aprilgrid).
- ORBIT_DIST: mean eye distance in m; rescales DISTANCE_FACTORS (0.5/0.9 x board width).
- ORBIT_FLIP=1: negates the board normal, so every eye sits on the BACK face. The camera still looks
  at the centre (make_pose), so it sees the texture mirror-reversed: v6 got 2 tags in 1266 frames.
- AIM_SCALE: multiplies every yaw/pitch in AIM_KB4/AIM_DS.  ARM_REACH: clamps eyes into a +-R m box.
- The centre "fix" (tx,ty,tz) is WRONG: the Quad mesh has local z=2.5 (engine.py:184) and sz=0.5
  (engine.py:244), so the board is drawn 1.25 m from (tx,ty,tz) along local +z; vertex mean was right.

## Board assembly limits
- A board reads only from its normal side (normal = quad local -z, R*(0,0,-1); no back-face culling,
  so the back shows the texture mirror-reversed and tag16h5 never decodes). v6 (ORBIT_FLIP=1): 100% of
  poses behind all 3 boards (off14 plane -0.47..-1.55 m) -> 2 tags/1266 CamD frames although tags were
  46-172 px with 0.5 px edges; the same frames h-flipped in software -> 47815 tags, all 3 grids. 2x
  upscaling gave 0. v5 (no flip): 100% of poses in front of off14 (25410 tags) but only 33%/9% in front
  of the tilted base/off15 boards (2276/18 tags). "In front" means on the normal side, not just ahead.
- Detector floor (v5 CamD frames shrunk in software, offline_tags_all.json): detection rate 50% at
  20 px bbox span (16 px black-square side), 90% at 37 px (25 px side), 95% at 47 px (30 px side).
  Real device p1 spans: CamD 44 px, KB4 31 px native. Plan for >= 30 px side = 30*d/f: at 1/3/5/7 m
  CamD f615 needs 4.9/14.6/24.4/34.1 cm tags, KB4 f395 needs 7.6/22.8/38.0/53.2 cm. Grid-0 strays in
  v6 are 10-20 px slivers on margins, not tags. Texture ids == config. Keep boards offset sideways so
  they do not stack on the view line (v2: off14 hid base 0-6).

## Do not break
Default launch, with no PLAYGROUND_BOARDS, must give exactly one 4x7 board
and the working build-trajectory flow. That is the user's known-good path.

## Camera models: what production uses
- CamD is Double Sphere. CamA, CamB, CamC are KB4. This is what the device
  file holds and what vk_calibrate must be given:
  `--cam-types kb4 kb4 kb4 ds --focal-lengths -1 -1 -1 550`
- Render CamD through the Double Sphere setting in the playground's Render
  dropdown. Do not render it as KB4.
- KB4 on CamD was a one-time diagnostic in August 2026. It isolated a
  Double Sphere focal/xi degeneracy on a single board plane by removing xi
  from the model. It is not the target and must not be reported as a pass.
- Double Sphere needs several board depths in one capture. With one board
  plane, xi trades against focal length and the fit returns about 3.4 percent
  focal error. The multi-board scene exists for this reason.
- Truth is the device file keyed by serial. Synthetic renders and the Aug 17
  real capture both use DP180IP-30020104.json, serial 30.02.0104.
Extrinsics on the same v9 solve, relative to CamD: rotation error 0.049 / 0.024 /
0.032 deg for CamA/B/C, position error 0.47 / 0.11 / 0.13 mm. Device file stores
translation in CENTIMETRES and an empty rotationMatrix for the reference camera.
Per-grid --tag-sizes confirmed correct: a wrong size would scale these baselines.
