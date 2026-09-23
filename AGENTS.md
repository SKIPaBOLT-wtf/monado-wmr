# Runtime agent guidance

Read `README.md`, `doc/g2-stability/STATUS.md`, `doc/g2-stability/PROVENANCE.md` and the current companion [WMR-Linux continuation prompt](https://github.com/SKIPaBOLT-wtf/WMR-Linux/blob/g2-stability/prompts/CONTINUE.md) before changing behavior. The companion project coordinates installation profiles, current hardware evidence and release gates. If its prompt path has moved, locate the current continuation document rather than applying a stale pasted snapshot blindly.

Preserve the upstream Git history and component licenses. Keep public changes free of device serials, factory/Windows calibration payloads, raw camera captures, room maps, account configuration, private email, local machine paths and debug artifacts. Use synthetic inputs in public tests. Windows reference disks remain read-only.

The initial branch contains bounded correctness fixes and experimental tracking behavior. Do not claim tracking acceptance from a hanging headset, an idle recording, feature counts, successful compilation or synthetic tests alone. Preserve working display/presentation and controller input while isolating a causal tracking change. Record coordinate frames, units, timestamp domains, world generations, validity and fallback behavior for every transform/fusion change.

Builds do not authorize installation. Do not run a default build preset that installs system-wide. Follow `doc/g2-stability/BUILD-AND-TEST.md`; use per-user staged artifacts and reversible rollback in the companion project. Do not install rejected Basalt rebuilds or enable the optional gyro-bias consumer without backend qualification. Do not acquire GL mirror textures on the affected SteamVR/Xwayland stack. CPU pose/property inspection is the established diagnostic route.

At each completed iteration update the status, what was tested, what remains unverified and exact source/build provenance. Future state takes precedence over this initial import record. Use English for code, documentation, issues and handoffs.
