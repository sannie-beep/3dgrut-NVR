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
6. "Not Fisheye" next to a Select button is **not** a bug and does not mean
   the DS parameters failed to load. `_draw_single_vk_cam` is called only for
   the SELECTED camera (`ps_gui.py` ~1465: the `if psim.Button(...)` fires
   only on the click frame, the `elif self.selected_camera_idx == idx` covers
   the rest). Unselected cameras draw a bare button with no text at all, and
   `psim.SameLine()` puts the selected camera's text on its own button's row.
   `selected_camera_idx` defaults to 0, so at startup the only text on screen
   is CamA's — and "Not Fisheye" is correct for CamA, whose DS slot is zeros.
   Verified by loading `VilotaDevice` directly: CamD is idx 3 with
   `distortionCoeff[5] = 431.2301`, so its condition is True and it reads
   "xi: -0.3, alpha: 0.55" once actually selected.
   Treat "Not Fisheye" at startup as the visible tell that **CamA is
   selected** — the state that silently downgraded every DS render (bug 6).

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
all four names. Subset renders with `only_cams` instead. Now `None` (all
four). Never trim `cam_names` itself.

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

### 6. Distortion indexed by the GUI selection, not the camera — FIXED (ps_gui.py:529)
`add_cam_to_vid_recorder(self, index)` took intrinsics from `index` but
distortion from `self.selected_camera_idx`:
```python
distortion_coefficients=self.distortions[self.selected_camera_idx] ...  # was
distortion_coefficients=self.distortions[index] ...                     # is
```
Every multi-camera export therefore paired each camera's own intrinsics with
whatever camera the GUI happened to have selected — by default index 0, CamA.
A four-camera render would have given all four CamA's k. Found because the
headless driver leaves `selected_camera_idx` as None, which turned the silent
wrong-coefficient path into `ValueError: Camera must have distortion
coefficients for rendering`. In the GUI it would never have raised.

This is a third way a CamD-only result could look fine while the rig was
wrong, so keep the pre-flight check in the driver: it asserts every camera's
fx and k[0:4] against `vk180.json` before a frame is rendered.

The MCAP export path no longer reads `selected_camera_idx`. The LIVE canvas
preview still does (`ps_gui.py:208`, `update_render_view_viz`), which is
correct — it previews the selected camera. So clicking Select CamD still
changes what you see on screen; it no longer changes what gets exported.

### Search range units — checked, not a bug
The live code builds `linspace(0, pi, steps=int(pi/step_size))`: 0..pi rad,
correct. The commented `theta_range = (0.0, 180.0)` at line 266 is dead and
never read. Do not wire it in — as radians it would span 10313 deg. Note
`step_size` sets the node COUNT, not the spacing; actual spacing is 0.001001
rad, about 0.395 px at CamA fx.

## Results

### Four-camera KB4 render → KB4 fit, 2026-08-26 — PASS, distortion validated
Post-fix (87bae80 + 60bcdf5 + the ps_gui.py:529 fix). Dataset
`mcap_outputs/orbit_4cam_kb4fix.mcap`, tags `orbit_4cam_kb4fix_tags.mcap`.
Per-camera orbit, 1296 poses, 1295 frames per camera, 71378 detections.

`success = true` for all four cameras. 0 of 621/949/953/1221 poses rejected.
Mean reprojection 0.414 px (cama 0.409, camb 0.419, camc 0.418, camd 0.410).

| cam | fx err | fy err | cx err | cy err | fov gate |
|---|---|---|---|---|---|
| CamA | +0.043% | +0.014% | -0.493 | -0.809 | 146 >= 130 |
| CamB | +0.001% | -0.033% | -0.452 | -0.457 | 142 >= 129 |
| CamC | -0.004% | -0.037% | -0.538 | -0.489 | 146 >= 130 |
| CamD | -0.020% | -0.047% | -0.485 | -0.467 | 202 >= 184 |

**Distortion now matches the file — this is the result bug 1 was hiding.**

| cam | k1 | k2 | k3 | k4 |
|---|---|---|---|---|
| CamA | -0.15% | +1.64% | +0.77% | +0.65% |
| CamB | -0.14% | -0.11% | -0.36% | -0.70% |
| CamC | -0.17% | +1.40% | +0.32% | +0.00% |
| CamD | (see below) | -12.6% | +0.81% | +2.46% |

CamA k1 recovers 0.3721 against the file's 0.372647. Pre-fix the same camera
returned -1.8e-4. CamD's k1 is -3.03e-05 in the file and +2.19e-05 recovered:
a sign flip on a number that is essentially zero, so the relative error is
meaningless — CamD is nearly equidistant and k3 (0.81%) is its dominant term.

This is the first run that validates the distortion path. Combined with the
earlier focal/principal-point/geometry/detector/solver evidence, the KB4
synthetic loop is now end-to-end.

The cx/cy offsets are the known half-pixel convention effect, unchanged.

#### Board coverage, per-camera orbit
| cam | frames | zero-tag | zero % | detections | mean/frame | >=12 tags |
|---|---|---|---|---|---|---|
| cama | 1265 | 513 | 40.6% | 9013 | 7.12 | 26.6% |
| camb | 1265 | 209 | 16.5% | 17316 | 13.69 | 58.5% |
| camc | 1265 | 211 | 16.7% | 17477 | 13.82 | 58.8% |
| camd | 1266 | 19 | 1.5% | 27572 | 21.78 | 93.3% |

CamA is the weak one at 40.6% zero-tag; CamB/CamC are fine at ~16.5%. It
still converged, but CamA has the fewest retained poses (621 vs 949/953/1221)
and the largest fx error. If CamA needs tightening, aim its grid, not the
whole orbit.

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

### Double Sphere single-plane — 3.4% focal bias — INVALID, NOT a degeneracy
**Retracted 2026-08-26. The DS runs were never Double Sphere renders. They
were equidistant renders, and the "bias" is a DS model fitted to an
equidistant image. Do not put this in the report as a degeneracy result.**

Proof, from `tags_extreme.mcap` alone (the recorded best run's detections,
no re-render needed) — the same corners fitted with two models:

| fit | focal | k / xi | reproj | success |
|---|---|---|---|---|
| ds | 446.008 (+3.43%) | xi -0.277063, alpha 0.56218 | — | false, fov 186<190 |
| kb4 | **614.820** | k = [-2.8e-04, 2.7e-04, -1.1e-04, 1.3e-05] | 0.4 px | true, fov 206 |

The kb4 fit lands 0.004% from the file's KB4 focal 614.8465 with k ~ 0.
That is the signature of an equidistant image. A genuine DS render cannot
look like this:

| render fitted with KB4 | f | k1 |
|---|---|---|
| true Double Sphere | 617.328 | -1.60e-02 |
| equidistant | 614.846 | ~0 |
| **tags_extreme (actual)** | **614.820** | **-2.82e-04** |

k1 is 57x smaller than the DS signature. The render was equidistant.

And the observed DS numbers are predicted by fitting DS to an equidistant
f=614.85 image over a ~190 deg field: f 445.20, xi -0.2784, alpha 0.5617,
against the observed 446.008, -0.277063, 0.56218. All three parameters.

Cause: with `camera_type = 'Double Sphere Fisheye'` the dispatch
(engine.py:1443) tests `distortion_coefficients[5] == 0.0` FIRST and routes
to `_raygen_kb4`. CamA's DS slot is all zeros, so bug 6 handing the renderer
CamA's coefficients silently downgraded the DS render to KB4, which bug 1
then made equidistant. The dropdown being left on KB4 (GUI trap 3) produces
the identical image, so the data cannot separate the two. Either way the
render was not DS.

What this retracts:
- the 3.4% figure as a measurement of anything about Double Sphere;
- "xi trades against focal length, one board at one depth cannot separate
  them" — plausible, but this experiment never tested it;
- the +/-0.3 px residual wave in `~/plot_aa.json` — that is the
  equidistant-vs-DS mismatch, not a degeneracy signature;
- the eliminated-causes list (trajectory, aim, distance, scene content,
  antialiasing, pixel grid, texture geometry). The eliminations are sound but
  they were chasing a bias with a different cause.

What survives: the projection-formula check (playground unproject vs basalt
project, exact inverses to 5 decimals) is a unit test of the maths and stands.
`generate_fisheye_rays_double_sphere` has still never been validated by a
render — no DS image on disk was produced by it.

### Real-device control
`real_device.mcap`, 674 frames. DS single-camera step 0 lands 0.48% off, and
the full run reproduces the device file to 0.03% (432.19 vs 432.07). This is
a device recording, so it never touched the playground renderer and none of
bugs 1/4/6 apply. It validates the solver's DS path. It says nothing about
the renderer's DS path, which remains untested.

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
- Four-camera pipeline (topics and config differ from the CamD-only one):
```bash
python fix_mcap_labels.py mcap_outputs/<name>.mcap <name>_img.mcap \
  --topics S1/cama S1/camb S1/camc S1/camd --serial DP180IP-2404-0004
CONFIG=offline_tags_all.json \
TOPIC="S1/cama/tags:queued S1/camb/tags:queued S1/camc/tags:queued S1/camd/tags:queued" \
  ./run_offline_tags.sh <name>_img.mcap <name>_tags.mcap
~/vk_src/vk_calibrate/build/vk_calibrate --vbag-path <name>_tags.mcap \
  --cam-types kb4 kb4 kb4 kb4 \
  --focal-lengths 395.21 396.92 395.67 614.85 --tag-size 0.15
```
  `offline_tags_all.json` is the four-camera detector config; the CamD-only
  `offline_tags_camd.json` silently yields one topic. Timings on this box:
  render 4x1295 frames ~11 min (KB4 cams ~13 it/s, CamD ~5 it/s), relabel 34 s,
  detection ~1 min, solve ~2 s. MCAP is ~3.4 GB (chunk-compressed).

## Multi-board blueprint (motivation now unconfirmed)
The degeneracy this was meant to fix has not actually been observed — see the
retracted DS result. Re-measure DS with a correct render first; only build the
multi-board scene if a real bias survives.

The real rig config `/opt/vilota/configs/camera_driver/vk180_calibration.json`
declares six grids: 1x3, 2x2, and four 7x4 boards with start_ids [0],
[0,14], [0,15], [0,16]. `generateGrid` expands multi-element start_ids
round-robin by increment ([0,14] → 0,14,1,15,...), which is how boards share
the tag family yet hash to distinct grid IDs. Reproducing this needs:
`create_aprilgrid.py` extended to accept a starting tag ID (currently it
always draws 0-27), four+ quads at different depths, and matching
`april_grids` entries in the detector config.

## Open work, in order
1. DONE 2026-08-26. `estimate_theta_star` verified, typo fixed (87bae80),
   override removed (60bcdf5), distortion indexing fixed (ps_gui.py:529),
   four-camera KB4 re-render and re-fit PASS with k matching the file.
   Every render before 60bcdf5 is equidistant — regenerate, never reuse.
   Still open: the bug 5 clamp/eps commit.
2. Re-measure Double Sphere. The recorded 3.4% is retracted (equidistant
   render, not DS). Needs a fresh CamD render with the dropdown on Double
   Sphere and CamD's own coefficients reaching the renderer — the pre-flight
   in `drive_render.py` checks the latter. Then fit `--cam-types ds`. If it
   returns ~431.2 the renderer's DS path is validated and there is no
   degeneracy to fix; only if a real bias survives is the multi-board scene
   worth building.
3. The report. Ingredients on disk: real-device control, three-model table
   (kb4 pass / radtan8 fail / ds fail), residual curve, eliminated-causes
   list, the two GUI traps, and the passing KB4 run.

## Rendering without the GUI
`ps_gui.py` drives everything from imgui buttons, so a scripted run has to call
the same methods in the same order. Working driver: `drive_render.py` in the repo root. Sequence:
`Playground(...)` -> `novel_view_renderer.load_device()` -> quad
`transform.reset(); sx=1.410; sy=0.825` -> `engine.camera_type = 'KB4'` ->
`video_recorder.frames_between_cameras = 1` -> `build_orbit_trajectory(pg)` ->
`pg.render_mcap_trajectory(poses)`. Needs `DISPLAY=:1` — polyscope still opens
a real GL window; it just never needs a click. `render_mcap_trajectory` ends
in `sys.exit()`.

The driver asserts before rendering, which is what turns a 12-minute mistake
into an instant one: quad wider than tall, override gone, `theta**3` present,
`only_cams` None, and every camera's fx/k[0:4] against `vk180.json`.

`threedgrut_playground/utils/orbit_trajectory.py` is byte-identical to
`~/orbit_percam.py.bak` — the "Build Orbit Trajectory" button already builds
the PER-CAMERA orbit (aim grids per camera type, 180 poses each for CamA/B/C
and 756 for CamD), not the old CamD-aimed one. Coverage numbers above are for
this orbit.

## Conventions
- **Never start a render. Renders are the user's to run.** If a step needs
  one, list the GUI steps and stop. The user runs it and says when the MCAP
  is written. Claude handles code, tests, the detector, the solver, and
  analysis — not the render itself. `drive_render.py` documents the call
  sequence and is useful for reading the settings off; do not execute it.
- Never guess an intrinsic value. Read it from `vk180.json`.
- Before every render check: quad sized (wider than tall), dropdown, and
  `only_cams`. A wrong setting costs 12 minutes.
- Keep run outputs under distinct names. Never overwrite a passing dataset.
- Distinguish "self-consistent round trip" from "faithful simulation" in all
  claims. Bug 1 makes several results the former only.
