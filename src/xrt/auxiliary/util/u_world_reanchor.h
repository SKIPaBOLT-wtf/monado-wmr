// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief World re-anchor glide: presentation-side convergence policy for SLAM
 *        relocalization / reset discontinuities ("head resnap guard", B2).
 *
 * A SLAM relocalization re-anchors the world in a single sample (worst recorded:
 * 819 mm + 59.3 deg in one 66.8 ms interval while the head was near-still). This
 * policy keeps one world-frame correction delta = (dp in R^3, dq in SO(3)),
 * applied at presentation time as
 *
 *     presented_q = dq * raw_q,   presented_p = raw_p + dp,
 *
 * and decaying to identity, so presentation is always SLAM truth composed with a
 * shrinking, rate-bounded world offset. Convergence to the tracker's new truth is
 * structural: delta -> identity implies presented == raw.
 *
 * Detection is innovation-based (Basalt exposes no covariance/confidence/state):
 * per new raw sample, the step beyond an IMU-derived envelope
 *
 *     ori_env = env_gyro_gain * integral(|gyro|) dt + ori_floor
 *     pos_env = max(env_speed * dt, pos_floor)
 *
 * is absorbed into delta (excess only — a zero-excess sample leaves delta
 * untouched, so clean segments are BIT-IDENTICAL to the raw stream and normal
 * jitter / real motion pass through with zero added latency; slow drift is
 * sub-envelope by definition and is never hidden). The glide is a rate-capped
 * exponential with motion-adaptive caps (a correction hides inside optic flow
 * when the head moves), a sub-perceptual snap to identity, and a sanity cap
 * beyond which the remainder passes through instantly (unwinding more than
 * ~90 deg / 1.5 m reads as vection and is worse than the cut it replaces).
 *
 * The reference implementation and the parameter derivation live in
 * results/b2-resnap-design-20260704/ (design.md + sim/sim_core.py, validated on
 * all 54 recorded resnap events of capture 20260703-202811); this C port is
 * checked against that simulator on the recorded head streams by
 * tests_world_reanchor. Angles in the API are DEGREES to match the simulator.
 */

#pragma once

#include <stdbool.h>

#include "xrt/xrt_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * Tunables for the world re-anchor glide. The defaults are the B2 design's
 * recommended point (sweep over 54 combos; all major-event convergence <= 0.6 s,
 * zero clean-segment absorption, minimal worst-event presented step): the worst
 * recorded event presents 298 mm / 6.8 deg instead of 819 mm / 59.3 deg and
 * converges in 0.50 s. Compile-time constants by design — the offline simulator
 * is the re-tuning harness, not an env knob.
 */
struct u_world_reanchor_params
{
	double tau_s;            //!< glide time constant (exponential regime)
	double omega0_dps;       //!< base glide angular-rate cap, deg/s
	double v0_mps;           //!< base glide linear-rate cap, m/s
	double beta_w;           //!< angular cap gain per deg/s of head angular speed
	double beta_v;           //!< linear cap gain per m/s of head speed
	double ori_floor_deg;    //!< orientation envelope floor per interval
	double snap_ang_deg;     //!< sub-perceptual snap threshold (angle)
	double snap_pos_m;       //!< sub-perceptual snap threshold (position)
	double cap_ang_deg;      //!< sanity cap on |delta| angle; excess passes through
	double cap_pos_m;        //!< sanity cap on |delta| translation
	double env_gyro_gain;    //!< envelope gain on the integrated gyro angle
	double env_speed_mps;    //!< position envelope speed
	double env_pos_floor_m;  //!< position envelope floor per interval
};

//! B2 recommended parameters (tau 0.10 s, caps 60 dps + 2|gyro| / 1.0 m/s + 1|v|,
//! floors 0.35 deg / 8 mm, snap 0.05 deg / 2 mm, sanity 90 deg / 1.5 m).
extern const struct u_world_reanchor_params u_world_reanchor_default_params;

//! The world correction delta. Doubles: the policy is validated bit-for-bit
//! against the float64 reference simulator, and delta must decay smoothly over
//! many small steps.
struct u_world_reanchor
{
	double dq[4]; //!< world correction rotation, quat xyzw
	double dp[3]; //!< world correction translation, m
	bool active;  //!< false <=> delta is exactly identity (presented == raw, bit-identical)
};

//! Innovation of one raw step against the IMU envelope. All angles deg, world frame.
struct u_world_reanchor_step
{
	double dang_deg;      //!< raw step rotation angle
	double dnorm_m;       //!< raw step translation norm
	double axis[3];       //!< raw step rotation axis (unit, world), valid when dang > 0
	double unit_dvec[3];  //!< raw step translation direction (unit, world), valid when dnorm > 0
	double exc_ang_deg;   //!< rotation excess beyond the envelope (0 = within)
	double exc_pos_m;     //!< translation excess beyond the envelope (0 = within)
};

void
u_world_reanchor_init(struct u_world_reanchor *wr);

/*!
 * Pure envelope test: innovation of raw step @p prev -> @p next over @p dt_s
 * given the integrated gyro angle @p gyro_int_deg for the same interval.
 * Returns true iff any excess (the step is a detected world re-anchor).
 * Shared by the presentation guard and the ESKF world-prior re-anchor so both
 * sites classify with the same math.
 */
bool
u_world_reanchor_compute_excess(const struct u_world_reanchor_params *prm,
                                const struct xrt_pose *prev,
                                const struct xrt_pose *next,
                                double dt_s,
                                double gyro_int_deg,
                                struct u_world_reanchor_step *out_step);

/*!
 * Build the rigid world transform Delta (x' = Delta.q * x + Delta.p) the
 * detected excess implies, with the rotation pivoted at @p pivot (the pre-step
 * head position): the transform that maps the pre-step head pose onto the
 * post-step head pose restricted to the excess — the best rigid estimate of the
 * SLAM world re-anchor. Applied to a world point at the head it reduces to the
 * guard's own (dp, dq) composition; applied to controller state it preserves
 * head-relative geometry, which is what lands the ESKF prior on the post-step
 * optical observations.
 */
void
u_world_reanchor_step_to_world_delta(const struct u_world_reanchor_step *step,
                                     const struct xrt_vec3 *pivot,
                                     struct xrt_pose *out_delta);

/*!
 * One policy update for a new raw sample: detect (envelope), absorb the excess
 * into delta, then decay delta over the interval with motion-adaptive rate caps.
 * @p gyro_int_deg integrated |gyro| angle over the interval (envelope input);
 * @p gyro_dps and @p speed_mps are the head angular / linear speed at the new
 * sample for the adaptive caps — IMU/tracker-state derived, NEVER a raw-position
 * finite difference (the step being absorbed would contaminate it and self-
 * inflate the cap at exactly the wrong moment; design §3 surprise 1).
 * Returns true iff any excess was absorbed this step; @p out_abs_ang_deg /
 * @p out_abs_pos_m (optional) receive the absorbed magnitudes.
 */
bool
u_world_reanchor_update(struct u_world_reanchor *wr,
                        const struct u_world_reanchor_params *prm,
                        const struct xrt_pose *raw_prev,
                        const struct xrt_pose *raw_new,
                        double dt_s,
                        double gyro_int_deg,
                        double gyro_dps,
                        double speed_mps,
                        double *out_abs_ang_deg,
                        double *out_abs_pos_m);

/*!
 * presented = delta composed with @p raw (orientation pre-rotated, position
 * shifted). When delta is identity this is an exact copy — presented output is
 * bit-identical to the raw pose on every clean frame.
 */
void
u_world_reanchor_apply(const struct u_world_reanchor *wr, const struct xrt_pose *raw, struct xrt_pose *out_presented);

//! Current delta magnitudes (deg, m). Zero when inactive.
void
u_world_reanchor_get_magnitude(const struct u_world_reanchor *wr, double *out_ang_deg, double *out_pos_m);

/* A controller presentation correction paired with its raw filter snapshot. dp is the
 * displacement at pivot, not the translation part of the rigid transform. */
struct u_world_reanchor_compensation
{
    double dq[4];
    double dp[3];
    double pivot[3];
    int64_t anchor_ns;
    double angular_cap_dps;
    double linear_cap_mps;
    uint64_t generation;
    bool active;
};

void
u_world_reanchor_compensation_init(struct u_world_reanchor_compensation *state);

/* Pure absolute-time evaluation: never advances mutable state on a pose pull. */
void
u_world_reanchor_compensation_evaluate(const struct u_world_reanchor_compensation *state,
                                      int64_t when_ns, struct xrt_pose *out_from_raw);

/* One exact raw-world rebase, right-composed inversely into the current presentation
 * correction. The same publication timestamp, pivot and rates go to both controllers.
 * Returns false for nonfinite/invalid or stale events; callers must then not rebase raw state. */
bool
u_world_reanchor_compensation_rebase(struct u_world_reanchor_compensation *state,
                                    const struct xrt_pose *raw_delta,
                                    const struct xrt_vec3 *new_raw_pivot,
                                    int64_t publication_ns, double gyro_dps, double speed_mps);

#ifdef __cplusplus
}
#endif
