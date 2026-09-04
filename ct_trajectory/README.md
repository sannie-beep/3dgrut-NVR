# ct_trajectory — continuous-time trajectory and synthetic IMU

Turns a list of poses into a function of time, then differentiates it
into a gyroscope and accelerometer stream. Covers steps 1 to 4 of the
workflow. Steps 5 (MCAP) and 6 (Basalt) are not here yet.

Requires numpy only. No torch, no scipy.

## Run the checks

```bash
source ~/niel_gs/3dgrut/.venv/bin/activate
python test_ct_trajectory.py
```

Expected output, in short:

| check | what it proves | result |
|---|---|---|
| 1 | analytic omega and accel match finite differences | 2e-10 rad/s, 1e-5 m/s^2 |
| 2 | Basalt Eqs. 10-12 integrate the IMU back onto the spline | error falls 10x per 10x rate |
| 3 | a device at rest reports 0 and R^T (0,0,+9.81) | exact |
| 4 | a fake orbit is 96% still with 53 m/s^2 spikes | as predicted |

Check 2 does not reach 1e-6 at 200 Hz. That is correct. Basalt's
scheme is first order in dt, so the error at 200 Hz is about 2 mm over
0.1 s and shrinks tenfold each time the rate rises tenfold. The
convergence is the proof, not the single number.

## Plug in the real orbit

```python
import numpy as np
from ct_trajectory import spline_from_matrices, sample_poses, synthesize_imu, gap_report

T = ...                                   # (N, 4, 4) body-to-world, from orbit_trajectory.py
print(gap_report(T[:, :3, 3]))            # expect frac_still near 0.96

sp = spline_from_matrices(T, duration=60.0, dt=0.1, timing="uniform")
cam_t, cam_T = sample_poses(sp, 30)       # for the renderer
imu_t, gyro, accel = synthesize_imu(sp, 200)
```

`T` must be body-to-world. If the orbit writes world-to-camera, invert
it first. If the scene is not z-up, pass `gravity=` to `synthesize_imu`.

## Design notes

Cumulative cubic B-spline, uniform knots, split interpolation on
SO(3) x R^3 (Hyperion Sec. 3.2, Eqs. 4-6). Rotation increments are
body-frame, right-multiplied. Angular velocity is closed form:

    omega_b = (A2 A3)^T l1' d1 + A3^T l2' d2 + l3' d3

where A_j = exp(l_j d_j), l_j are the cumulative basis functions and
d_j = log(R_{j-1}^T R_j).

The spline approximates control points rather than passing through
them. For IMU synthesis that does not matter. For rendering the
waypoints exactly, swap in a Z-spline later.

## Known gaps

- `timing="arclength"` exists but is untested on real data. With 96%
  repeated positions it will pile almost all the time onto the hops.
  A travel trajectory is the real fix.
- No noise. Deliberate. Add it after step 6 works.
- No MCAP writer. Topic name and schema depend on which Basalt reads it.
