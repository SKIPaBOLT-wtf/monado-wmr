# Build and test this branch

Builds here do not install anything, start SteamVR or access the headset. Use the dependencies described in the upstream README. The reference environment has a C/C++ compiler, CMake, Ninja, Python 3, Eigen3, OpenCV and the Linux Monado dependencies available. The preset explicitly requests WMR, SLAM, SteamVR and tests; it still performs dependency detection. It is not a containerized, bit-reproducible release toolchain.

```sh
cmake --preset g2-stability
cmake --build --preset g2-stability --parallel 4
python3 tests/g2/test_hid_preservation.py
python3 tests/g2/test_bias_binding.py
python3 tests/g2/test_mask_safety.py
ctest --test-dir build/g2-stability --output-on-failure -R 'kalman_fusion|world_reanchor|wmr_config|wmr_protocol|wmr_camera_footer'
ctest --test-dir build/g2-stability --output-on-failure -R '^tests_slam_feature_support$'
```

The driver output is `build/g2-stability/steamvr-monado/bin/linux64/driver_monado.so`. Its matching resource tree is `build/g2-stability/steamvr-monado/resources`. The preset builds the driver and selected existing source tests. It deliberately does not invoke the upstream `default` build preset, whose target is `install`.

`test_hid_preservation.py` and `test_bias_binding.py` need only Python and `cc`/`g++`; they extract functions from this checkout and compile synthetic fixtures in temporary directories. HID tests cover both firmware channels, full sensor reports, original receive times, deferred lifecycle events, packet size/echo validation and absolute deadlines. Bias tests cover optional ABI rejection/fallback, immutable timestamp association, accepted-history binding, missing static calibration, invalid observations and reset behavior. Two deliberate code mutations must fail.

`test_mask_safety.py` additionally links the freshly built runtime's pose math without initializing VR. Pass `--build PATH` for another build directory. It tests source-extracted mask freshness and world-cache rebasing with synthetic pinhole bounds; its historical fixture must demonstrate the old defects. It covers the 150 ms boundary, future times, missing history, fresh idle, capping, behind-camera geometry and full rigid-world invariance. It is not a camera detector accuracy test.

With `SLAM_WRITE_CSVS=true`, the driver also writes `feature-support.csv` alongside `tracking.csv` at the same raw pose timestamps. Each camera reports whether the feature extension was available, its count, its finite projected `u/v` count, and the major/minor RMS axes in the backend's `u/v` units. Equal counts can have very different spatial spread; the focused test checks that distinction. These are projected-landmark summaries, not reprojection residuals, static-scene labels, independent ground truth, or a threshold for accepting a pose. Keep diagnostic CSVs private because even aggregated scene geometry can describe a room. Normal launch keeps CSV recording disabled.

The existing `tests_kalman_fusion` target includes physics-derived synthetic motion tests. Public source tests do not need private camera captures. Large private identical-input replays remain valuable for deployment qualification, but must not be uploaded by default.

A successful compile and synthetic tests are necessary, not sufficient, for live release. Require coordinated head/controller motion, ordinary wearer micro-movement, restart/reconnect, loss/reacquisition and sustained scene tests before marking tracking accepted. A resting headset alone cannot establish comfort or pose accuracy.

Do not automatically replace Basalt while building this driver. The current integration uses the retained upstream release used as the comparison baseline (overall tracking remains unaccepted). The original GNU ld source-control and optional bias-export libraries failed identical-input qualification, even though the ABI tests passed. Keep `G2_PREDICT_WITH_VIT_BIAS=false` until a replacement backend passes broader qualification.

The failed libraries were linked with GNU ld. The companion `2026-09-23-basalt-linker-parity` iteration relinked their identical compiled objects with mold 2.40.4 and matched all 2,553 release rows on the original failing input. That isolates a linker selection difference on this replay, but does not qualify an installed replacement or learned bias. The companion's read-only `wmrctl basalt-linker` check rejects silent fallback to GNU ld; a complete-input replay remains required.
