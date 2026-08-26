# Vilota 3DGRUT playground — synthetic camera calibration

## Goal
Internship project. Render a known camera (Vilota VK180, four cameras) from a
Gaussian splat scene, detect AprilTags in the render with Vilota's own
detector, feed the detections to Vilota's calibration tool, and check that it
returns the intrinsics the calibration file already holds. A written report
for the team lead is a deliverable.

## Environment
- Repo: `~/niel_gs/vilota-ref` (Vilota fork of NVIDIA 3DGRUT playground)
- venv: `source ~/niel_gs/3dgrut/.venv/bin/activate`
- Launch: `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python playground.py --gs_object ply_files/room.ply`
- Solver: `~/vk_src/vk_calibrate/build/vk_calibrate` (outside the repo)
- Vilota runtime: `vk_camera_driver`, `vk_record`; capnp schemas in
  `/opt/vilota/messages`; real detector configs in
  `/opt/vilota/configs/camera_driver/`
- Device serial used throughout: `DP180IP-2404-0004`
- Backups (verify before trusting): `~/vilota_fixes.patch`,
  `~/orbit_percam.py.bak`, `~/plot_real.json`, `~/plot_aa.json`

## The camera file: `calibration_files/vk180.json`
CamA/B/C are type 1, 1280x800, KB4, substantial barrel distortion:
- cam0 fx 395.21 fy 394.92, k = [0.372647, 0.046181, -0.121431, 0.040402]
- cam1 fx 396.92 fy 396.55, k = [0.354893, 0.079276, -0.148230, 0.048361]
- cam2 fx 395.67 fy 395.06, k = [0.363951, 0.046050, -0.118613, 0.038761]

CamD is type 0, 1920x1200, the fisheye, and the file describes it twice:
- KB4 (`intrinsicMatrix` + `distortionCoeff[0:4]`): 614.846, 613.596,
  939.427, 585.253, k = [-3e-05, 0.000252, 0.000714, -9.4e-05]
- Double Sphere (`distortionCoeff[5:11]`): 431.2301, 430.4876, 939.6711,
  585.3367, xi -0.3016, alpha 0.5530

Both entries describe the same lens. The DS entry converts to a best-fit
equidistant focal near 617.3, which is 0.4% from 614.85. Each ray generator
reads the entry that matches its own model: KB4 reads
`camera.get_camera_intrinsics()` plus `distortionCoeff[0:4]`, DS reads
`distortionCoeff[5:11]`. This is correct, not an inconsistency.

## GUI traps (each one has cost a render)
1. The board quad starts SQUARE. The size field shows 15.00 at startup, but
   the quad keeps its own shape until you press
   **Primitives → Quad 1 → Transform → Reset to 15cm square size**.
   The texture is 752x440 (aspect 1.7091). On a square quad, tags render at
   w/h near 0.585 and every calibration result is garbage (the fy/fx = 1.76
   incident). Check by eye: the board must be visibly wider than tall.
2. Do NOT use the **Square size (cm)** button. `calculate_sx_and_sy`
   (`ps_gui.py` ~line 1300) swaps rows and columns, so it returns the exact
   inverse of the correct sx/sy.
3. The render dropdown defaults to Double Sphere. For a KB4 run, set it to
   KB4 before pressing render.
4. The GUI writes every export to `mcap_outputs/long_final_path.mcap`.
   Rename it before the next render overwrites it.
5. Ctrl+click **Frames Between** to type a value. Keep it at 1.

## Known code bugs

### 1. `fisheye.py` `theta_star = ru` debug override — FIXED 2026-08-26 (60bcdf5)
```python
theta_star = estimate_theta_star(k1, k2, k3, k4, ru=ru, ...)
theta_star = ru   # debug override, discarded the correction
```
Made every KB4 render equidistant. Removed. Ray-level check after the fix:
CamA half-width now solves to 1.14538 rad (131.2 deg across), matching the
file's coefficients; the equidistant render was 1.6194 rad (185.6 deg), about
55 deg too wide. CamD half-width 1.54878 vs 1.5614 equidistant. Emitted ray
angles agree with a bisection inverse of the KB4 polynomial to ~1e-7 rad.

Removing this alone would NOT have been correct — see bug 4, which it masked.

### 2. `threedgrut_playground/ps_gui.py` lines ~1465-1472
`cam_names.index(cam_name)` selects the extrinsics, so `cam_names` must keep
all four names. Subset renders with `only_cams` instead. Currently
`["CamD"]`. Set to `None` for all four. Never trim `cam_names` itself.

### 3. `calculate_sx_and_sy` rows/cols swap (GUI trap 2 above).

### 4. `estimate_theta_star` inverted the wrong polynomial — FIXED 2026-08-26 (87bae80)
```python
d = lambda theta: theta + k1 * theta**2 + ...   # was
d = lambda theta: theta + k1 * theta**3 + ...   # is
```
KB4 uses odd powers 3, 5, 7, 9. Terms k2/k3/k4 were right; the k1 term had
exponent 2. Round-trip error (theta -> ru via true KB4 -> estimate_theta_star
-> theta), before -> after:

| cam | before | after |
|---|---|---|
| CamA | 1.07e-01 rad (42.3 px) | 1.7e-07 rad |
| CamB | 9.46e-02 rad (37.6 px) | 1.9e-07 rad |
| CamC | 1.11e-01 rad (43.8 px) | 1.6e-07 rad |
| CamD | 7.39e-05 rad (0.05 px)  | 2.1e-08 rad |

**Bug 1 masked bug 4.** The override threw away `estimate_theta_star`'s
return value, so the typo never reached a render and no result on this page
could have exposed it. Anyone who had deleted the override without reading
the polynomial would have traded a 36 deg corner error on CamA for a 42 px
one, and the residual would have looked plausible: the error is NOT monotonic
in radius — it peaks near 22 px around 37 deg, crosses zero near 58 deg, then
climbs to 99 px at the corner. Fix both or neither.

CamD's k1 is -3e-05, so CamD alone would never have shown this. CamA/B/C
k1 ~ 0.36 is where it bites. Another reason not to validate on CamD only.

Verified by `tests/test_estimate_theta_star.py` (round trip, search-range
units, past-lens-limit behaviour). Its section 2 is a regression guard: it
feeds ru built from the OLD theta^2 polynomial, so a near-zero error there
means the typo is back.

### 5. LUT extrapolates past its end; no eps on the ru divide — OPEN
`estimate_theta_star` clamps `idx` but not the interpolation, so ru beyond
`R[-1]` is linearly extrapolated off the table (ru = 1.5x R[-1] -> 189 deg,
10x -> 348 deg). Negative ru extrapolates below zero the same way.
`generate_rays_kb4` divides by ru (`m_x / ru`) with no eps, so a pixel exactly
on the principal point gives NaN, and its docstring promises an
`out_of_fov_mask` it does not return. Latent, not active: max in-frame ru is
1.91 (CamA corner) against R[-1] ~ 866. CamD's polynomial does turn over at
theta 2.868 rad, past its 1.81 rad corner, so the table is only valid below
that. Deferred to a third commit.

### Search range units — checked, not a bug
The live code builds `linspace(0, pi, steps=int(pi/step_size))`: 0..pi rad,
correct. The commented `theta_range = (0.0, 180.0)` at line 266 is dead and
never read. Do not wire it in — as radians it would span 10313 deg. Note
`step_size` sets the node COUNT, not the spacing; actual spacing is 0.001001
rad, about 0.395 px at CamA fx.

## Results

### KB4 render → KB4 fit, CamD only, 2026-08-26 — PASS, but SUPERSEDED
**Obtained with the bug 1 override ACTIVE (pre-60bcdf5), so the render was
equidistant. The recovered k are not comparable to any post-fix run — do not
diff them against a new result, and do not treat the k mismatch below as an
open question. Re-run this before it goes in the report.** The focal,
principal point, geometry, detector and solver conclusions still stand.

614.752 613.337 938.953 584.772
k = -2.03e-04 2.11e-04 -9.43e-05 1.36e-05
success = true, fov 206.00 >= 187, calibrated_r 1105.0 px (ratio 0.976)
0 of 1222 poses rejected, ~0.47 px reprojection

Focal error 0.015%. First `success = true` of the project. The wide-diversity
orbit moved `calibrated_r` from 996.2 (stuck across nine configurations) to
1105.0. The recovered k do NOT match the file (k3 off by 10x, sign flip),
which is exactly what bug 1 predicts: the render was equidistant, and a KB4
fit of an equidistant image is exact at k = 0. So this validated geometry,
detector, solver, and focal/principal-point recovery, and nothing about the
distortion path. Not end-to-end validation. The post-fix re-run is the one
that tests distortion.

### Double Sphere on single-plane synthetic data — 3.4% focal bias
Best run (`tags_extreme`): 446.0 vs truth 431.2, xi -0.2771 vs -0.3016,
reprojection 0.46 px, fov gate fails at 186 < 190. Cause: xi trades against
focal length, and one board at one depth cannot separate them. The residual
vs angle is a smooth ±0.3 px wave (zero crossings ~41/69/87 deg, saved in
`~/plot_aa.json`) — the signature of the fitted DS disagreeing with the true
DS, consistent with the degeneracy. Eliminated by measurement: trajectory,
aim, distance, scene content (splats on/off), antialiasing (1x vs 4x
identical to 3 decimals), pixel grid convention, texture geometry, and the
projection formula itself (matches Usenko et al. 2018; playground unproject
and basalt project are exact inverses to 5 decimals).

### Real-device control
`real_device.mcap`, 674 frames. DS single-camera step 0 lands 0.48% off, and
the full run reproduces the device file to 0.03% (432.19 vs 432.07). The
real rig uses six boards (see blueprint below), which breaks the degeneracy.

### CamA/B/C synthetic (four-camera bundle)
SUPERSEDED — pre-60bcdf5, equidistant render. Focal correct (395.27 vs
395.21), k1 near -1.8e-4 vs the file's 0.3726. That was bug 1, not the
solver, and CamA is where bug 4 would have bitten hardest too (42 px).
Re-run after re-rendering. Single-vs-multi camera was tested and is NOT
the cause of the DS bias. Coverage note: the CamD-aimed orbit leaves CamA/B
with ~45% zero-tag frames; a four-camera run needs the per-camera trajectory
(`~/orbit_percam.py.bak`).

### Constant half-pixel principal point offset — explained, harmless
The playground casts rays at pixel centers; AprilTag reports corners half a
pixel away. Absorbed into cx/cy (~0.47 px low both axes). Report it, do not
chase it.

## Working pipeline (after a render exits)
```bash
mv mcap_outputs/long_final_path.mcap mcap_outputs/<name>.mcap
python fix_mcap_labels.py mcap_outputs/<name>.mcap <name>_img.mcap \
  --topics S1/camd --serial DP180IP-2404-0004
TOPIC="S1/camd/tags:queued" ./run_offline_tags.sh <name>_img.mcap <name>_tags.mcap
~/vk_src/vk_calibrate/build/vk_calibrate --vbag-path <name>_tags.mcap \
  --cam-types kb4 --focal-lengths 616 --tag-size 0.15
```
- `fix_mcap_labels.py` fixes two labels only: channel encoding (image/jpeg →
  capnp Image) and the zero `step` field. Pixels are untouched yuv420.
- `run_offline_tags.sh` chains vk_camera_driver playback + vk_record, config
  `offline_tags_camd.json` (tag16h5, black_border 2, 7x4).
- Config `size` is 0.05173 but the rendered tags are 0.15 m. That scales
  translations, not intrinsics. `--tag-size 0.15` on the solver is what
  matters.
- Useful solver flags: `--cam-types kb4 kb4 kb4 ds`, `-n "cama" "camb" ...`
  to exclude cameras, `--bypass-dataset-check`.
- The Streamlit UI's Monocular Fisheye preset produces `--cam-types ds` for
  CamD, never kb4. Monocular Pinhole untested. Call the binary directly.
- `measure_tags.py <png> --show out.png` measures tag squareness on frames
  in `ecal_pub_sub/CamD/`.
- `python tests/test_estimate_theta_star.py` — KB4 inverse round trip. No GPU,
  no render, ~1 s. Run it after any edit to `fisheye.py`.

## Multi-board blueprint (for the DS degeneracy fix)
The real rig config `/opt/vilota/configs/camera_driver/vk180_calibration.json`
declares six grids: 1x3, 2x2, and four 7x4 boards with start_ids [0],
[0,14], [0,15], [0,16]. `generateGrid` expands multi-element start_ids
round-robin by increment ([0,14] → 0,14,1,15,...), which is how boards share
the tag family yet hash to distinct grid IDs. Reproducing this needs:
`create_aprilgrid.py` extended to accept a starting tag ID (currently it
always draws 0-27), four+ quads at different depths, and matching
`april_grids` entries in the detector config.

## Open work, in order
1. DONE 2026-08-26: `estimate_theta_star` verified, typo fixed (87bae80),
   override removed (60bcdf5). NOT yet re-rendered. Next: re-render and re-fit
   CamD (k must now match the file), then all four cameras with the per-camera
   orbit. Every render before 60bcdf5 is equidistant — regenerate, do not
   reuse. Optional third commit: the bug 5 clamp/eps.
2. Multi-board scene, then retry Double Sphere single-camera.
3. The report. Ingredients on disk: real-device control, three-model table
   (kb4 pass / radtan8 fail / ds fail), residual curve, eliminated-causes
   list, the two GUI traps, and the passing KB4 run.

## Conventions
- Never guess an intrinsic value. Read it from `vk180.json`.
- Before every render check: quad sized (wider than tall), dropdown, and
  `only_cams`. A wrong setting costs 12 minutes.
- Keep run outputs under distinct names. Never overwrite a passing dataset.
- Distinguish "self-consistent round trip" from "faithful simulation" in all
  claims. Bug 1 makes several results the former only.
