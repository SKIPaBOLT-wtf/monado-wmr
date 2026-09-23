# Monado WMR — G2 stability work

Experimental native Linux tracking and SteamVR driver work for HP Reverb G2 and related Windows Mixed Reality hardware. This branch preserves [AshishKumar4/monado-wmr](https://github.com/AshishKumar4/monado-wmr) and the upstream [Monado project](https://monado.freedesktop.org/), with additional transport, timing, coordinate-frame and validity fixes.

**Display and game rendering work on the reference system. Head and controller tracking quality is still under investigation. This is not a release claiming Windows-equivalent tracking or general hardware support.** The G2 uses inside-out cameras and IMUs; it does not require base stations.

The companion project, [WMR-Linux](https://github.com/SKIPaBOLT-wtf/WMR-Linux/tree/g2-stability), owns setup, hardware profiles, public status, the detailed continuation prompt and packaging work. This repository owns the runtime source and its source-level regressions.

- [Build and test](doc/g2-stability/BUILD-AND-TEST.md)
- [Status and regression boundaries](doc/g2-stability/STATUS.md)
- [Source and asset provenance](doc/g2-stability/PROVENANCE.md)
- [Upstream README and dependency guidance](doc/g2-stability/UPSTREAM-README.md)
- [Contributing to Monado](CONTRIBUTING.md)

Please report exact versions and a redacted diagnostic summary. Do not attach raw room images, factory calibration, device serials, Steam account configuration, registry hives or crash cores to a public issue. See the companion project's privacy guidance before sharing evidence.
