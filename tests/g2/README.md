# G2 source regressions

See [build and test](../../doc/g2-stability/BUILD-AND-TEST.md). These tests use synthetic input and production function extraction. Generated programs live in temporary directories and do not contact the headset, SteamVR, Windows disks or a network. No private room captures or calibration are required.

The two small source-only suites run in GitHub Actions. The mask suite also requires a built driver for its production pose math. Existing CMake test targets provide broader runtime coverage. Positive physical tracking acceptance is a separate release gate.
