# Runtime status at the initial stability import

This record is a dated starting point (2026-09-23), not a permanent claim about future branches. Read the companion project's current state before continuing work.

## Preserved behavior

The source retains the G2 display/90 Hz and native SteamVR path and public controller model resources. NVIDIA presentation/kernel workarounds belong to the companion integration and must remain hardware/version gated. This runtime import does not change the machine's installed driver, display settings, game configuration or kernel.

## Fixes with bounded evidence

- HMD HID reports previously disappeared during synchronous controller firmware reads. Full reports now dispatch with original timestamps while lifecycle events use a bounded deferred FIFO. Source fixtures and a live IMU cadence check support this fix.
- Historical controller-prior queries, reanchor epochs, optical acceptance, IEKF residual sign and idle sentinel handling have source-level corrections. They do not establish overall controller tracking accuracy.
- Dead-reckoning history boundaries return finite untracked fallback rather than integrating missing history backward. Pose draining and reset/getter state are serialized.
- Controller frontend optical caches now follow the same rigid-world change as the estimator. Untracked masks expire using the existing 150 ms optical freshness horizon. Synthetic full-SE3 tests establish frame invariance; the most recent live capture did not exercise a populated controller cache.
- Zero current visual observations can produce invalid tracking rather than a trusted-looking extrapolation. Positive observation counts are not a complete quality metric.
- A gyro-only optional VIT bias consumer binds bias to the exact accepted pose timestamp and factory-calibration path. It is compiled but disabled by default because the rebuilt backend is not qualified.

## Unresolved

1. HMD raw visual-inertial position can still drift or diverge even with positive feature observations and no enabled controller masks. Long-term visual anchoring, map/relocalization behavior, bias estimation and bad-observation rejection still need causal evidence.
2. Moving the head can move apparently stationary controllers. Several timing and world-frame defects were corrected, but combined moving-head behavior has not passed user acceptance.
3. Micro-jitter, fast-stop settling and perceived prediction lag remain unresolved. Arbitrary smoothing can trade visible noise for latency and must not replace a timestamp/frame/calibration diagnosis.
4. Controller aim/grip/SteamVR Home alignment remains physically unverified. Manufacturer/public profile transforms and application bindings are evidence; ad hoc inward/downward adjustments are experiments, not factory truth.
5. A room/floor origin can correct a constant room placement error, but cannot repair raw visual-inertial drift. Do not hide a drifting tracker with repeated recentering.
6. The Windows investigation compared specific calibration and driver evidence; it did not reproduce all proprietary Microsoft fusion/filtering mechanisms.
7. The GNU ld Basalt rebuilds diverged on a recording where the installed upstream release remained bounded. In the companion `2026-09-23-basalt-linker-parity` iteration, mold 2.40.4 relinks of the same compiled objects matched all 2,553 release rows on that input. The old GNU ld artifacts remain rejected, and the mold-linked test outputs are not deployment-qualified. Do not enable the optional gyro-bias consumer from this offline result.

## Default policy

The reference integration uses two HMD SLAM cameras, camera auto-exposure, a qualified Basalt release, zero-observation validity checks enabled, gyro bias extension disabled and Windows HT1 extrinsics override disabled. Generic hardware profiles must perform capability checks; do not apply the reference policy to an untested headset merely because it is WMR.

No GL mirror-texture acquisition during diagnostic work: it has crashed this SteamVR/Xwayland stack. Use CPU pose/property APIs and offline logs instead. Do not write to a Windows partition while using its installation as a reference.
