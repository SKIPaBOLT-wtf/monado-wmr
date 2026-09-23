// Copyright 2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Implementation of LED constellation tracking
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#include <inttypes.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "os/os_threading.h"

#include "tracking/t_led_models.h"
#include "tracking/t_constellation_tracking.h"

#include "util/u_debug.h"
#include "util/u_frame.h"
#include "util/u_g2_telemetry.h"
#include "util/u_logging.h"
#include "util/u_sink.h"
#include "util/u_trace_marker.h"
#include "util/u_var.h"
#include "util/u_worker.h"
#include "util/u_world_reanchor.h"

#include "internal/blobwatch.h"
#include "internal/association_hypothesis.h"
#include "internal/camera_model.h"
#include "internal/correspondence_search.h"
#include "internal/debug_draw.h"
#include "internal/joint_contention.h"
#include "internal/joint_pnp.h"
#include "internal/l2_accept_gate.h"
#include "internal/l1_depth_verdict.h"
#include "internal/multicam_triangulate.h"
#include "internal/ransac_pnp.h"
#include "internal/sample.h"
#include "internal/static_map.h"
#include "internal/work_budget.h"
#include "internal/yaw_belief.h"

DEBUG_GET_ONCE_LOG_OPTION(ct_log, "CONSTELLATION_LOG", U_LOGGING_INFO)

#define ASSOC_COLD_SEARCH_STARTING_WORKERS 2
#define ASSOC_COLD_SEARCH_THREADS 4

/* Deterministic cold-search work-unit budget (1 unit = 1 P3P trial, pose check = 5 units; see
 * correspondence_search.c). ASSOC_FRAME_WORK_BUDGET caps the TOTAL cold-search work per frame
 * across devices+views. Sized for the G2's 22.2ms median camera period: 40k units x 228ns =
 * 9.1ms serial worst case, leaving headroom under the 22ms frame budget for blob extraction
 * (p99 2ms) and the uncharged warm paths (~5ms). The capture-replayed sweep on the band-2
 * clutter capture measured fast p95 13.3ms with 2/4029 frames >22ms at this cap, vs p95 77ms /
 * 673 frames >22ms unbudgeted — and the MAX_FAST_QUEUE_SIZE=2 sink queue stops dropping frames.
 * Every allowance is PRE-ASSIGNED from frame content + tracker state before any search task
 * runs (never a shared pool drained in completion order), so identical inputs spend identical
 * units live and offline. The units->ms mapping is machine-specific (228ns is this CPU);
 * the budget stays deterministic everywhere and the drift is observable via the
 * G2_TELEM_EV_TRACKER_WORK_UNITS event + the search stream's work_spent field. */
#define ASSOC_FRAME_WORK_BUDGET 40000u
/* The rotating full-frame deep-view slot gets twice a shallow-only view's share: the measured
 * success-trials p50 for full-frame deep passes (46.1k) is ~2x full-frame shallow (22.5k). */
#define ASSOC_COLD_DEEP_VIEW_WEIGHT 2u
/* Per-view allowance for the uncovered-view pickup of tracked devices (wave D): 5x the p50 of
 * today's bounded pickup spend (512 trials) and well under its 9216 deep-tail, so the common
 * pickup is unchanged while the worst case stays bounded now that BOUNDED_SEARCH is gone. */
#define ASSOC_COLD_UNCOVERED_VIEW_ALLOWANCE 2560u

#define MIN_ROT_ERROR DEG_TO_RAD(30)
#define MIN_POS_ERROR 0.10

/* Covariance-driven prior-consistency gate: size the prior tolerance by the fusion's LIVE 1-sigma
 * uncertainty (PRIOR_GATE_SIGMA sigmas), instead of a fixed value. MIN_*_ERROR above are the FLOORS
 * (so when the fusion is confident the gate is no looser, and no tighter, than before -> no
 * regression; flips are rejected by the large rotation error they produce regardless). MAX_*_ERROR
 * are the CEILINGS that bound how far the gate widens after an optical dropout, when the fusion
	 * inflates its covariance -> projected-prior and cold-search hypotheses are scored against a realistic
	 * envelope instead of a stale tight prior. One statistical knob (the sigma multiplier); the rest are
	 * physical floors/ceilings. */
#define PRIOR_GATE_SIGMA 3.0 /* ~99.7% per axis */
#define MAX_POS_ERROR 0.60
#define MAX_ROT_ERROR DEG_TO_RAD(60)

/* B5 SLAM controller-mask repair (results/b5-ledmask-20260704/design.md). The masks SLAM must
 * ignore are pushed at FRAME cadence from the live prediction + last-seen pose (mirroring the
 * static-map exemption pattern), replacing the accept-only push whose rects went pixel-frozen
 * for seconds during head rotation (age p90 ~3 s / max 10 s, masking up to 86% of a camera and
 * stealing 10-45% of the scarce dark-frame feature pool). Masking controller light is
 * load-bearing divergence protection (unmasked light: repeatable early reset + 90 m-1.7 km
 * post-reset divergence in the 6+6 causal A/B), so coverage must continue through coast — the
 * prediction rect is inflated by the projected PRIOR_GATE_SIGMA position uncertainty and a
 * prediction the fusion cannot bound (giant rect) self-disables via the PER-DEVICE area cap
 * instead of masking a wrong region (an unpredictable controller is overwhelmingly outside the
 * SLAM cams' view too). MASK_HALO_MARGIN_PX closes the measured halo leak at the rect edges
 * (LED glow extends 2-3x beyond the bounding rect; 0.59% of strong dark corners leaked there).
 * MASK_AREA_CAP_FRACTION is enforced PER DEVICE, not on the per-camera total: the design
 * sketch's total-sum cap measured a leak REGRESSION on the 20260703-202811 emulation (dark
 * leak 1.50% vs 0.59% live) because it rationed two HONEST close-range rects against each
 * other, unmasking an in-view controller — exactly the harm class masking exists to prevent.
 * Per-device at the same fraction passes both acceptance gates (dark leak 0.45% < 0.59% live;
 * masked non-controller theft 3.8% vs 9.3% live; results/b5-ledmask-20260704/sweep_cap.py)
 * and keeps >=80% of every camera maskable-free per device, far below the measured harm
 * regime (45-86% stale-rect blankets). */
#define MASK_HALO_MARGIN_PX 20.0f
#define MASK_AREA_CAP_FRACTION 0.20f
/* Floor for the depth used to project the position sigma to pixels: a controller at/behind the
 * camera plane projects an unbounded sigma; the area cap then disables the rect. */
#define MASK_SIGMA_MIN_DEPTH_M 0.05f

/* Tilt scale for the soft anisotropic mirror-flip / prior-orientation cost. The mirror twin of a few-LED
 * PnP almost always TILTS the controller wrong, and gravity is driftless: the fusion prior's TILT is
 * gravity-anchored (the ESKF keeps anchoring from the controller's accel even through an optical dropout)
 * so the prior tilt is a valid reference even when the yaw prior is stale. The scale is the LIVE fusion
 * horizontal-plane 1-sigma (see get_pose_uncertainty tilt_std) clamped to
 * [FLIP_COST_TILT_SIGMA_MIN, GRAVITY_TILT_TOL]:
 *   - GRAVITY_TILT_TOL (30 deg = the rotation floor MIN_ROT_ERROR, the legacy fixed scale) is now the
 *     CEILING — the worst case during violent dynamics when the accel residual widens the tilt covariance.
 *     A fixed 30-deg sigma everywhere was unjustified physics for a gravity-observed DoF: it charged a
 *     36.7-deg tilt twin only ~1.5 nats, letting a 5-blob single-view accept scrape through at a 0.2-nat
 *     margin (xv1 cam1_150739624107385; results/h6-rmodel-20260612).
 *   - FLIP_COST_TILT_SIGMA_MIN mirrors FLIP_COST_YAW_SIGMA_MIN's over-confidence guard: the covariance
 *     cannot see optical/LED-model/timing error, so the scale never claims more tilt confidence than the
 *     measured honest-accept envelope. Measured across the four validation captures (54k accepted fits):
 *     honest tilt-vs-prior error p50 0.5 deg / p90 2.8 / p99 8.5 -> at a 5-deg sigma the honest p99 sits
 *     at 1.7 sigma (quadratic zone, ~3 nats) while a 36.7-deg tilt twin lands 7.3 sigma (Huber-linear,
 *     ~35 nats) — decisively rejected instead of margin-threaded. */
#define GRAVITY_TILT_TOL MIN_ROT_ERROR
#define FLIP_COST_TILT_SIGMA_MIN DEG_TO_RAD(5)

/* The mirror-flip disambiguation is a SOFT cost re-rank, not a hard veto: for each candidate pose
 * and its mirror twin, cost = reproj_error_px + pose_metrics_prior_orient_cost(...), and the LOWEST-cost
 * candidate is committed — a frame is never dropped for ambiguity ("fix, don't reject"). The cost is the
 * anisotropic squared Mahalanobis distance of the candidate orientation from the prior — tilt scaled by
 * the tight driftless GRAVITY_TILT_TOL, yaw by the live fusion 1-sigma — Huber-robustified and weighted
 * into px. A confident prior makes the prior term dominate (a tilt flip's huge distance is never selected),
 * while an uncertain/untracked prior makes the yaw term vanish and reprojection decide. The
 * Huber knee reuses the 3-sigma envelope (PRIOR_GATE_SIGMA): within it the penalty is quadratic, beyond
 * it linear, so a gross outlier (flip) cannot dominate pathologically. FLIP_COST_WEIGHT commensurates the
 * dimensionless robustified distance with the per-LED reprojection error (px^2). FLIP_COST_YAW_SIGMA_MAX
 * is the untracked/long-dropout yaw scale CEILING — at 180deg it lets a 180deg flip cost d^2=1.0
 * (negligible), so twins tie on reprojection during fast motion. 90deg keeps the legitimate-motion
 * relaxation while still penalising flips at d^2=4. */
#define FLIP_COST_WEIGHT 1.0
#define FLIP_COST_HUBER_KNEE_SIGMA PRIOR_GATE_SIGMA
#define FLIP_COST_YAW_SIGMA_MAX DEG_TO_RAD(90)
/* The yaw-scale FLOOR for the soft flip cost: the minimum yaw 1-sigma the cost will use, regardless of how
 * confident the ESKF reports it is. This is a separate concept from the GRAVITY_TILT_TOL tilt scale (they are
 * different DoFs and must not share one constant), but the VALUE here is governed by a real effect the ESKF
 * yaw covariance cannot see: the optical front-end occasionally folds a wrong-yaw pose, after which the filter
 * is legitimately CONFIDENT (small reported yaw sigma) in a yaw that is actually off — an over-confident prior.
 * Measured on G2 controller captures, ~11-19% of accepted optical poses disagree with the gyro/fusion prior by
 * >45deg, so a yaw scale tighter than this floor makes the cost over-trust that stale prior and SELECT the
 * candidate that matches the wrong yaw (offline A/B: dropping the floor to 3deg raised Left wrong-branch
 * 24.8->37.0% and flip 10.0->11.3%, while 30-60deg are statistically flat). The floor caps that over-trust;
 * the live sigma still WIDENS the scale above it after a real dropout (the cost relaxes and reprojection
 * decides). Equal to MIN_ROT_ERROR, the prior gate's per-axis rotation floor: the yaw scale never claims more
 * yaw confidence than the gate's tightest accepted rotation tolerance, the empirically validated knee. */
#define FLIP_COST_YAW_SIGMA_MIN MIN_ROT_ERROR

/* Yaw-sigma above which a 4-LED single-cam PnP defers: above this, the soft prior cost can no longer
 * reliably tell the mirror twins apart. 5+ LEDs break near-coplanarity so no twin is emitted. */
#define FLIP_COST_YAW_SIGMA_GUARD DEG_TO_RAD(60)

/* Head-anchored yaw cue (Fix A): the soft flip cost adds the candidate's HEAD-RELATIVE-yaw distance
 * from the last accepted controller-in-head orientation. INDEPENDENT of the world-frame fusion prior:
 * a 180-deg mirror flip flips head-relative yaw too, so this signal stays valid even when the world
 * prior is stale, and is robust to SLAM global drift (a delta in head frame cancels common drift).
 *
 * TTL caps how long after the last accept the cue is trusted: too long and head+hand have moved
 * independently enough that head-relative yaw is no longer a tight constraint; too short and we miss
 * the consecutive-frame disambiguation that is the whole point. ~100ms covers ~3 frames at 30Hz —
 * the regime in which the twin-disambiguation matters most. */
#define HEAD_YAW_CUE_TTL_NS (100ll * 1000ll * 1000ll)
/* Yaw scale for the head-anchored term: chosen so legitimate head-vs-controller relative motion (the
 * fastest plausible 30deg / frame at 30Hz, i.e. ~900deg/s relative rotation) sits within ~1 sigma
 * while a 180-deg flip is ~3 sigma -> huber-bent linear penalty. */
#define HEAD_YAW_CUE_YAW_SIGMA DEG_TO_RAD(60)
/* Tilt scale wide enough to NOT penalise legitimate hand-tilt-vs-head-tilt motion: this term is the
 * YAW signal; the world-frame term already handles the driftless tilt. */
#define HEAD_YAW_CUE_TILT_SIGMA DEG_TO_RAD(180)

/* Covariance-gated partial-fold. When the unified associator cannot lock a pose but the fusion has
 * a usable prior, we still fold the
 * individual LEDs CONFIDENTLY matched to the prior, each gated by the ESKF's anisotropic per-LED
 * innovation covariance S = H·P·Hᵀ + R (predict_led_gate). A blob<->LED pairing is folded iff its
 * Mahalanobis distance d² = rᵀS⁻¹r ≤ this χ²₂ quantile. χ²₂ inverse-CDF: -2·ln(1-p). p=0.99 -> 9.21,
 * matching the fold's own per-LED gate (CHI2_GATE_2DOF) so the two lines of defence are consistent.
 * The anisotropic S (tight tilt / loose yaw after a dropout) makes a tilt-flipped correspondence land
 * outside the gate automatically — no separate gravity check needed here. */
#define PARTIAL_FOLD_CHI2_2DOF 9.21 /* -2*ln(1-0.99) */
#define PARTIAL_TRI_MIN_LEDS 4
#define PARTIAL_TRI_MAX_STD_M 0.04f

#define CT_TRACE(c, ...) U_LOG_IFL_T(c->log_level, __VA_ARGS__)
#define CT_DEBUG(c, ...) U_LOG_IFL_D(c->log_level, __VA_ARGS__)
#define CT_INFO(c, ...) U_LOG_IFL_I(c->log_level, __VA_ARGS__)
#define CT_WARN(c, ...) U_LOG_IFL_W(c->log_level, __VA_ARGS__)
#define CT_ERROR(c, ...) U_LOG_IFL_E(c->log_level, __VA_ARGS__)

/* Maximum number of frames to permit waiting in the fast-processing queue */
#define MAX_FAST_QUEUE_SIZE 2

/* Predictive-ROI blob detection (Oasis driver convergent design, RE'd from MS's
 * ConnectedComponent::Locate -> ILedLocationPredictor::Predict path): for each tracked device, project
 * every LED in its constellation through the device's current ESKF state to its predicted image-pixel
 * position in the camera; the union of per-LED patches (each sized to that LED's actual prediction
 * uncertainty from predict_led_gate's S) gives a tight region the blob detector restricts its
 * flood-fill to. Eliminates ambient-noise blobs from the correspondence search (the matcher's
 * dominant "52% idle" defect on our side) and pre-localizes blobs around specific LED candidates.
 *
 * Per-LED adaptive sizing (NOT MS's fixed 16-px patch): each LED's pad = max(ROI_BASE_PAD_PX,
 * ROI_SIGMA_K * pixel_sigma), where pixel_sigma = sqrt(max diag of S). The S matrix returned by
 * predict_led_gate captures both the ESKF posterior position uncertainty AND the LED's geometric
 * sensitivity at this camera+pose. A precise prediction (small S) gets a near-minimum pad; an
 * uncertain prediction (large S, e.g. post-coast or low-confidence pose) gets a wider pad
 * proportionally. This per-LED scaling naturally handles asymmetric controllers (left vs right) and
 * coast-vs-fresh frames without a single global pad that must be conservative enough for the worst
 * case while also tight enough for the best.
 *
 * Fallback: when fewer than ROI_FALLBACK_MIN_PREDICTIONS LEDs project into the frame across all
 * tracked devices (cold start, lost tracking), the helper returns false and the caller uses the
 * full-frame path -- matching MS's PatchSearchFallbackMinPoints = 6.
 *
 * Always-on; the cold-start path falls through to full-frame via the predict_led_gate untracked
 * return. */
/* MS uses PredictivePatchSize/2 = 8 as their per-LED floor, but their predictor extrapolates the ESKF
 * state with IMU at the frame's EXPOSURE time, so their predictions are tight. Our predict_led_gate
 * uses the most-recent ESKF state which can lag the actual exposure by up to ~11ms (one frame at 90Hz);
 * empirically the temporal drift floor is much larger than 8px. Bisection on the headpose capture
 * (commit b8cc1fab2) found bp=32 with sigma_k=3.0 is the genuine sweet spot — 5pp wrongBr wins on
 * both controllers without crossing the threshold where larger pads admit ambient noise. The S matrix
 * scaling (per-LED) handles uncertainty growth above this floor. */
#define ROI_BASE_PAD_PX 32      /* floor: temporal-drift-aware; MS's 8 is too tight for our timing */
#define ROI_SIGMA_K 3.0f        /* per-LED pad = k * sqrt(max-diag(S)); 3σ ~ 99.7% coverage */
#define ROI_FALLBACK_MIN_PREDICTIONS 6
#define ROI_MAX_OPTICAL_AGE_MS 150.0

//! The OpenCV(+Y down, +Z away) <-> OpenXR(+Y up, +Z toward) camera-basis flip: a 180-deg rotation about
//! X (i.e. negate Y and Z). Single source for the convention so every CV<->XR conversion agrees.
static const struct xrt_pose P_YZ_FLIP = {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};

//! Sandwich a pose by the YZ flip (convert a pose between the OpenCV and OpenXR camera bases).
static void
pose_flip_YZ(const struct xrt_pose *in, struct xrt_pose *dest)
{
	struct xrt_pose tmp;
	math_pose_transform(&P_YZ_FLIP, in, &tmp);
	math_pose_transform(&tmp, &P_YZ_FLIP, dest);
}


/* Map an xrt_device type to the telemetry device_id (0=HMD, 1=left, 2=right). */
static uint8_t
telem_device_id(const struct xrt_device *xdev)
{
	if (xdev == NULL)
		return 0;
	switch (xdev->device_type) {
	case XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER: return 1;
	case XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER: return 2;
	default: return 0;
	}
}

/* Pack an xrt_pose into the [px,py,pz, qx,qy,qz,qw] layout telemetry expects. */
static void
telem_pack_pose(const struct xrt_pose *p, float out[7])
{
	out[0] = p->position.x;
	out[1] = p->position.y;
	out[2] = p->position.z;
	out[3] = p->orientation.x;
	out[4] = p->orientation.y;
	out[5] = p->orientation.z;
	out[6] = p->orientation.w;
}

enum g2_search_result
{
	G2_SEARCH_SUCCESS = 0,
	G2_SEARCH_NO_SEARCHABLE_ANCHORS = 1,
	G2_SEARCH_NO_ANCHOR_WITH_3_NEIGHBOURS = 2,
	G2_SEARCH_NO_P3P_TRIALS = 3,
	G2_SEARCH_NO_POSE_CHECKS = 4,
	G2_SEARCH_ALL_POSE_CHECKS_PRUNED = 5,
	G2_SEARCH_BEST_NOT_GOOD = 6,
	G2_SEARCH_NO_GOOD_CANDIDATE = 7,
	G2_SEARCH_ROI_SKIP_NO_BLOBS = 8,
	G2_SEARCH_ROI_SKIP_NO_MODEL = 9,
	G2_SEARCH_ROI_SKIP_UNTRUSTED_PRIOR = 10,
	G2_SEARCH_ROI_SKIP_INVALID_PRIOR = 11,
	G2_SEARCH_ROI_SKIP_VISIBLE_LT3 = 12,
	G2_SEARCH_ROI_SKIP_BLOBS_LT4 = 13,
	G2_SEARCH_ROI_SKIP_FULL_EQUIV = 14,
};

static enum g2_search_result
telem_classify_search_result(bool success, const struct correspondence_search_diagnostics *diag)
{
	if (success) {
		return G2_SEARCH_SUCCESS;
	}
	if (diag->searchable_anchors == 0) {
		return G2_SEARCH_NO_SEARCHABLE_ANCHORS;
	}
	if (diag->anchors_with_3_neighbours == 0) {
		return G2_SEARCH_NO_ANCHOR_WITH_3_NEIGHBOURS;
	}
	if (diag->num_trials == 0) {
		return G2_SEARCH_NO_P3P_TRIALS;
	}
	if (diag->num_pose_checks == 0) {
		return G2_SEARCH_NO_POSE_CHECKS;
	}
	if (diag->num_pose_checks_pruned >= diag->num_pose_checks) {
		return G2_SEARCH_ALL_POSE_CHECKS_PRUNED;
	}
	if (diag->best_any_blobs_matched > 0) {
		return G2_SEARCH_BEST_NOT_GOOD;
	}
	return G2_SEARCH_NO_GOOD_CANDIDATE;
}

static uint32_t
telem_search_bng_reason_flags(enum correspondence_search_flags flags,
                              enum g2_search_result result,
                              const struct correspondence_search_diagnostics *diag)
{
	if (result != G2_SEARCH_BEST_NOT_GOOD || diag == NULL) {
		return 0;
	}

	const uint32_t match_flags = diag->best_any_match_flags;
	const uint32_t matched = diag->best_any_blobs_matched;
	const uint32_t visible = diag->best_any_leds_visible;
	const uint32_t unmatched = diag->best_any_unmatched_blobs;
	const float reproj_per_match =
	    matched > 0 ? diag->best_any_reproj_err_px / (float)matched : INFINITY;
	const bool has_prior = (match_flags & POSE_HAD_PRIOR) != 0 ||
	                       (flags & CS_FLAG_HAVE_POSE_PRIOR) != 0;
	uint32_t reasons = 0;

	if (matched < 3) {
		reasons |= G2_SEARCH_BNG_MATCHED_LT3;
	}
	if (has_prior && (match_flags & POSE_MATCH_POSITION) == 0) {
		reasons |= G2_SEARCH_BNG_PRIOR_POSITION_FAIL;
	}
	if (has_prior && (match_flags & POSE_MATCH_ORIENT) == 0) {
		reasons |= G2_SEARCH_BNG_PRIOR_ORIENT_FAIL;
	}
	if ((match_flags & POSE_MATCH_LED_IDS) == 0) {
		reasons |= G2_SEARCH_BNG_LED_IDS_FAIL;
	}

	const bool clean_cluster = unmatched * 4 <= matched && matched >= 5;
	const bool covers_visible = 2 * visible <= 3 * matched;
	const bool minimal_prior = matched >= 4 && unmatched == 0 && (match_flags & POSE_MATCH_LED_IDS) != 0;
	const bool priorless_large = visible > 6 && matched > 6;

	if (has_prior) {
		if (!(reproj_per_match < 2.0f)) {
			reasons |= G2_SEARCH_BNG_REPROJ_FAIL;
		}
		if (!clean_cluster) {
			reasons |= G2_SEARCH_BNG_CLEAN_CLUSTER_FAIL;
		}
		if (!covers_visible) {
			reasons |= G2_SEARCH_BNG_VISIBLE_COVER_FAIL;
		}
		if (!minimal_prior) {
			reasons |= G2_SEARCH_BNG_MINIMAL_PRIOR_FAIL;
		}
		if (!priorless_large) {
			reasons |= G2_SEARCH_BNG_PRIORLESS_LARGE_FAIL;
		}
	} else {
		if (!priorless_large) {
			reasons |= G2_SEARCH_BNG_PRIORLESS_LARGE_FAIL;
		}
		if (!(reproj_per_match < 3.0f)) {
			reasons |= G2_SEARCH_BNG_REPROJ_FAIL;
		}
		if (!clean_cluster) {
			reasons |= G2_SEARCH_BNG_CLEAN_CLUSTER_FAIL;
		}
		if (!covers_visible) {
			reasons |= G2_SEARCH_BNG_VISIBLE_COVER_FAIL;
		}
	}

	return reasons;
}

static void
telem_emit_search_result(uint8_t device_id,
                         int view_id,
                         uint64_t timestamp_ns,
                         int pass,
                         enum correspondence_search_flags flags,
                         bool success,
                         bool prior_tilt_trusted,
                         const struct correspondence_search_diagnostics *diag)
{
	const enum g2_search_result result = telem_classify_search_result(success, diag);
	const uint32_t bng_reason_flags = telem_search_bng_reason_flags(flags, result, diag);
	const float reproj_per_match = diag->best_any_blobs_matched > 0
	                                   ? diag->best_any_reproj_err_px / (float)diag->best_any_blobs_matched
	                                   : INFINITY;
	const float unmatched_per_match = diag->best_any_blobs_matched > 0
	                                      ? (float)diag->best_any_unmatched_blobs /
	                                            (float)diag->best_any_blobs_matched
	                                      : INFINITY;
	const float matched_visible_ratio = diag->best_any_leds_visible > 0
	                                        ? (float)diag->best_any_blobs_matched /
	                                              (float)diag->best_any_leds_visible
	                                        : 0.0f;
	g2_telem_search(device_id, (uint8_t)view_id, timestamp_ns, (uint8_t)pass, (uint8_t)result,
	                (uint16_t)flags, prior_tilt_trusted ? 1 : 0, diag->input_blobs,
	                diag->searchable_anchors, diag->filtered_anchors, diag->anchors_with_3_neighbours, diag->neighbour_links,
	                diag->num_trials, diag->num_pose_checks, diag->num_pose_checks_pruned,
	                (uint8_t)diag->min_led_depth, (uint8_t)diag->max_led_depth,
	                (uint8_t)diag->max_blob_depth, (uint8_t)diag->best_any_pose_blob_depth,
	                (uint8_t)diag->best_any_pose_led_depth, diag->best_any_match_flags,
	                (uint8_t)diag->best_any_leds_visible, (uint8_t)diag->best_any_blobs_matched,
	                (uint8_t)diag->best_any_unmatched_blobs, diag->best_any_reproj_err_px,
	                bng_reason_flags, reproj_per_match, unmatched_per_match, matched_visible_ratio,
	                diag->work_spent, diag->budget_exhausted);
}

static void
telem_emit_search_skip(uint8_t device_id,
                       int view_id,
                       uint64_t timestamp_ns,
                       int pass,
                       enum g2_search_result result,
                       enum correspondence_search_flags flags,
                       bool prior_tilt_trusted,
                       uint32_t input_blobs,
                       uint32_t visible_leds,
                       uint32_t roi_blobs)
{
	g2_telem_search(device_id, (uint8_t)view_id, timestamp_ns, (uint8_t)pass, (uint8_t)result,
	                (uint16_t)flags, prior_tilt_trusted ? 1 : 0, input_blobs, visible_leds, roi_blobs, 0, 0, 0,
	                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0f, 0, 0.0f, 0.0f, 0.0f, 0, 0);
}

void
t_constellation_camera_group_dump_json(const struct t_constellation_camera_group *cams, FILE *f)
{
	if (cams == NULL || f == NULL) {
		return;
	}
	fprintf(f, "{\n");
	fprintf(f, "  \"format\": \"g2-constellation-cameras\",\n");
	fprintf(f, "  \"version\": 1,\n");
	fprintf(f, "  \"cam_count\": %d,\n", cams->cam_count);
	fprintf(f, "  \"ctrl_gain\": %u,\n", cams->ctrl_gain);
	fprintf(f, "  \"cameras\": [\n");
	for (int i = 0; i < cams->cam_count; i++) {
		const struct t_constellation_camera *c = &cams->cams[i];
		const struct t_camera_calibration *cal = &c->calibration;
		fprintf(f, "    {\n");
		fprintf(f, "      \"index\": %d,\n", i);
		fprintf(f, "      \"slam_tracking_index\": %zu,\n", c->slam_tracking_index);
		fprintf(f, "      \"width\": %d,\n", cal->image_size_pixels.w);
		fprintf(f, "      \"height\": %d,\n", cal->image_size_pixels.h);
		fprintf(f,
		        "      \"intrinsics\": [[%.12g,%.12g,%.12g],[%.12g,%.12g,%.12g],[%.12g,%.12g,%.12g]],\n",
		        cal->intrinsics[0][0], cal->intrinsics[0][1], cal->intrinsics[0][2], cal->intrinsics[1][0],
		        cal->intrinsics[1][1], cal->intrinsics[1][2], cal->intrinsics[2][0], cal->intrinsics[2][1],
		        cal->intrinsics[2][2]);
		fprintf(f, "      \"distortion_model\": \"%s\",\n",
		        t_stringify_camera_distortion_model(cal->distortion_model));
		fprintf(f, "      \"distortion\": [");
		for (int k = 0; k < XRT_DISTORTION_MAX_DIM; k++) {
			fprintf(f, "%s%.12g", k ? "," : "", cal->distortion_parameters_as_array[k]);
		}
		fprintf(f, "],\n");
		fprintf(f,
		        "      \"P_imu_cam\": {\"position\": [%.12g,%.12g,%.12g], "
		        "\"orientation\": [%.12g,%.12g,%.12g,%.12g]},\n",
		        c->P_imu_cam.position.x, c->P_imu_cam.position.y, c->P_imu_cam.position.z,
		        c->P_imu_cam.orientation.x, c->P_imu_cam.orientation.y, c->P_imu_cam.orientation.z,
		        c->P_imu_cam.orientation.w);
		fprintf(f, "      \"roi\": {\"x\": %d, \"y\": %d, \"w\": %d, \"h\": %d},\n", c->roi.offset.w,
		        c->roi.offset.h, c->roi.extent.w, c->roi.extent.h);
		fprintf(f, "      \"blob_min_threshold\": %u, \"min_threshold\": %u\n", c->blob_min_threshold,
		        c->min_threshold);
		fprintf(f, "    }%s\n", (i + 1 < cams->cam_count) ? "," : "");
	}
	fprintf(f, "  ]\n}\n");
}

struct t_constellation_tracked_device_connection
{
	/* Device and tracker each hold a reference to the connection.
	 * It's only cleaned up once both release it. */
	struct xrt_reference ref;

	/* Index in the devices array for this device */
	int id;

	/* Protect access when around API calls and disconnects */
	struct os_mutex lock;
	bool disconnected; /* Set to true once disconnect() is called */

	// Callbacks to the tracked device
	struct xrt_device *xdev;
	struct t_constellation_tracked_device_callbacks *cb;

	struct t_constellation_tracker *tracker; //! Parent tracker instance
};

struct constellation_tracker_device
{
	struct t_constellation_tracked_device_connection *connection;

	bool have_led_model;
	struct t_constellation_led_model led_model;
	struct t_constellation_search_model *search_led_model;

	bool have_last_seen_pose;
	uint64_t last_seen_pose_ts;
	struct xrt_pose last_seen_pose; // global pose
	int last_matched_blobs;
	int last_matched_cam;
	struct xrt_pose last_matched_cam_pose; // Camera-relative pose

	/* Head-anchored yaw cue for the soft mirror-flip cost (Fix A): the last accepted controller
	 * orientation expressed in the HMD/head IMU frame, and its timestamp. Used as a SECOND prior
	 * (alongside the world-frame fusion prior) when ranking mirror twins on the next solve: a 180-deg
	 * flip is reflected in both world and head-relative yaw, so adding the head-relative yaw distance
	 * roughly doubles flip-vs-legit discrimination at the matcher gate. Stale by HEAD_YAW_CUE_TTL_NS;
	 * silently skipped when no fresh accept is available. */
	bool have_last_head_rel_quat;
	uint64_t last_head_rel_quat_ts;
	struct xrt_quat last_head_rel_quat; // controller orientation in head/IMU frame

	/* Previous committed pose-lock orientation in the matcher's world frame. Used as a bounded temporal-yaw
	 * continuity term for mirror-twin ranking; position-only and LED-fold observations do not update it. */
	bool have_temporal_yaw_ref;
	uint64_t temporal_yaw_ref_ts;
	struct xrt_quat temporal_yaw_ref_quat;
	bool have_temporal_yaw_prior;
	struct xrt_quat temporal_yaw_ref_prior;

	/* Rotating full-frame deep-view slot (wave C): which qualifying view gets this device's
	 * deep cold pass, advanced once per frame the device runs the full-frame wave. Keyed on
	 * processed-frame count — never wall-clock — so a lost device completes full depth across
	 * all views within <= n_views frames and the rotation replays deterministically offline. */
	uint32_t cold_deep_view_rr;

	/* Fast-thread-confined (seeded, stepped and read only from the association pipeline on the
	 * fast tracking thread) — needs no lock; see association_{seed,update}_yaw_belief. */
	struct constellation_yaw_belief yaw_belief;
};

struct constellation_tracker_camera_state
{
	//! Distortion params
	struct camera_model camera_model;
	//! ROI in the full frame mosaic
	struct xrt_rect roi;
	//! Camera's pose relative to the HMD GENERIC_TRACKER_POSE (IMU)
	struct xrt_pose P_imu_cam;

	//! Constellation tracking blob extraction — fast-thread-confined (process, label updates and
	//! observation release all run on the fast tracking thread; no lock).
	blobwatch *bw;
	int last_num_blobs;

	//! Evidence-accumulating retention: world-anchored static-clutter map for this camera
	//! (fast thread only; updated right after blob extraction each frame).
	struct static_map static_map;

	//! Per-device cold-search state for the unified associator: each (camera, device) scope
	//! task owns one instance, and the waves are serialized by wait_all, so one per device
	//! suffices (a task runs its shallow and conditional deep pass sequentially on it).
	struct correspondence_search *cold_search[CONSTELLATION_MAX_DEVICES];

	//! Debug output
	struct u_sink_debug debug_sink;
	struct xrt_pose debug_last_pose;
	struct xrt_vec3 debug_last_gravity_vector;

	//! The index into the slam tracking camera array this camera represents
	size_t slam_tracking_index;
};

/* One cold-search scope = (device, view, blob set): the unit of work-budget pre-assignment.
 * A task runs the shallow pass and then — ONLY if its own shallow results produced no GOOD
 * pose, and only when it owns the deep slot — the deep pass, sequentially, within one
 * pre-assigned allowance. All outputs are task-local; merging into the shared per-device
 * hypothesis list happens after u_worker_group_wait_all in fixed (device, view) order, so
 * nothing about the spend or the results depends on pool completion order. */
struct association_cold_scope_task
{
	struct correspondence_search *cs;
	struct t_constellation_search_model *search_model;
	struct blob *blobs;
	int num_blobs;
	struct xrt_pose P_cam_obj;
	struct xrt_vec3 prior_pos_error;
	struct xrt_vec3 prior_rot_error;
	struct xrt_vec3 cam_gravity_vector;
	float prior_yaw_sigma_rad;
	float prior_tilt_sigma_rad;
	enum correspondence_search_flags pass_flags[2];
	uint32_t work_allowance;
	bool run_deep;
	uint64_t timestamp_ns;
	uint8_t device_id;
	int dev_slot;
	int view_id;
	int pass_base;
	bool prior_tilt_trusted;
	bool ignore_prior_results;

	/* Task-local outputs: [0] = shallow pass, [1] = conditional deep pass. */
	bool pass_ran[2];
	int n_results[2];
	struct correspondence_search_result results[2][CORRESPONDENCE_SEARCH_MAX_RESULTS];
	struct correspondence_search_diagnostics diag[2];
	uint32_t work_spent;

	/* Prior-ROI scopes search a blob subset; it must outlive the push onto the pool. */
	struct blob roi_blobs[MAX_BLOBS_PER_FRAME];
};

/*!
 * An @ref xrt_frame_sink that analyses video frame groups for LED constellation tracking
 * @implements xrt_frame_sink
 * @implements xrt_frame_node
 */
struct t_constellation_tracker
{
	//! Receive (mosaic) frames from the camera
	struct xrt_frame_sink base;
	//! frame node to insert in the xfctx
	struct xrt_frame_node node;

	/*! HMD device we get observation base poses from
	 * and that owns the xfctx keeping this node alive */
	struct xrt_device *hmd_xdev;

	struct os_mutex tracked_device_lock;

	//! Tracked device communication connections
	int num_devices;
	struct constellation_tracker_device devices[CONSTELLATION_MAX_DEVICES];

	//!< Tracking camera entries
	struct constellation_tracker_camera_state cam[XRT_TRACKING_MAX_SLAM_CAMS];
	int cam_count;
	//!< Commanded controller-slot analog gain shared by all cameras (t_constellation_camera_group::ctrl_gain);
	//!< 0 = unknown -> the gain-16 calibration point. Passed per frame into blobwatch (K gain law + telemetry).
	uint16_t ctrl_gain;

	/* Debug */
	enum u_logging_level log_level;
	bool debug_draw_normalise;
	bool debug_draw_blob_tint;
	bool debug_draw_blob_circles;
	bool debug_draw_blob_ids;
	bool debug_draw_blob_unique_ids;
	bool debug_draw_leds;
	bool debug_draw_prior_leds;
	bool debug_draw_last_leds;
	bool debug_draw_pose_bounds;
	bool debug_draw_device_bounds;

	uint64_t last_frame_timestamp;

	/*! World re-anchor detection (the head resnap guard's tracker-side complement): previous
	 * sampled HMD pose + |gyro| for the IMU-envelope innovation test on the very head-pose
	 * stream the camera extrinsics are composed from. Armed only while the sampled relation
	 * carries a VALID angular velocity — the envelope is IMU-relative, so without a gyro signal
	 * a re-anchor cannot be told from head motion (the offline replay's recorded head stream
	 * deliberately has none, keeping replays byte-identical by construction). */
	bool reanchor_have_prev;
	struct xrt_pose reanchor_prev_pose;
	int64_t reanchor_prev_ts;
	double reanchor_prev_gyro_dps;

	uint64_t last_fast_analysis_ms;
	uint64_t last_blob_analysis_ms;
	uint64_t last_assoc_work_units;

	// Fast tracking thread
	struct xrt_frame_sink *fast_q_sink;
	struct xrt_frame_sink fast_process_sink;
	struct u_worker_thread_pool *cold_search_pool;
	struct u_worker_group *cold_search_group;
	//! Scope-task storage for the cold-search waves (one frame in flight on the fast thread;
	//! each wave uses at most one task per (device, view) and waves are serialized).
	struct association_cold_scope_task cold_scope_tasks[CONSTELLATION_MAX_DEVICES * CONSTELLATION_MAX_CAMERAS];

	//! Frames fully processed through the pipeline. Lets the offline harness barrier on
	//! per-frame completion instead of racing a fixed sleep (debug/test only).
	atomic_uint_fast64_t frames_completed;

	struct xrt_device_masks_sample controller_masks_sample;
	struct xrt_device_masks_sink *controller_masks_sink;
};

static void
constellation_tracked_device_connection_notify_frame(struct t_constellation_tracked_device_connection *ctdc,
                                                     uint64_t frame_mono_ns,
                                                     uint64_t frame_sequence)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->notify_frame_received) {
		ctdc->cb->notify_frame_received(ctdc->xdev, frame_mono_ns, frame_sequence);
	}
	os_mutex_unlock(&ctdc->lock);
}

/* ctdc->lock is the raw-world publication barrier. Keep validation and each fold/cache callback
 * in one critical section; a frozen old-frame pose must never enter a newly rebased estimator. */
static bool
prior_epoch_matches_locked(struct t_constellation_tracked_device_connection *ctdc,
                           const struct t_estimator_prior *prior)
{
	return prior == NULL || (ctdc->cb->validate_prior_epoch &&
	                        ctdc->cb->validate_prior_epoch(ctdc->xdev, prior));
}

static bool
constellation_tracked_device_connection_prior_current(struct t_constellation_tracked_device_connection *ctdc,
                                                      const struct t_estimator_prior *prior)
{
	os_mutex_lock(&ctdc->lock);
	bool ret = !ctdc->disconnected && prior_epoch_matches_locked(ctdc, prior);
	os_mutex_unlock(&ctdc->lock);
	return ret;
}

static bool
constellation_tracked_device_connection_get_estimator_prior(struct t_constellation_tracked_device_connection *ctdc,
                                                            timepoint_ns when_ns, struct t_estimator_prior *out)
{
	if (!ctdc) { return false; }
	os_mutex_lock(&ctdc->lock);
	bool ret = !ctdc->disconnected && ctdc->cb->get_estimator_prior &&
	           ctdc->cb->get_estimator_prior(ctdc->xdev, when_ns, out);
	os_mutex_unlock(&ctdc->lock);
	return ret;
}

static void
constellation_tracked_device_connection_notify_pose(struct t_constellation_tracked_device_connection *ctdc,
                                                    timepoint_ns frame_mono_ns,
                                                    const struct t_estimator_prior *prior,
                                                    const struct xrt_pose *pose)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && prior_epoch_matches_locked(ctdc, prior) && ctdc->cb->push_observed_pose) {
		ctdc->cb->push_observed_pose(ctdc->xdev, frame_mono_ns, pose);
	}
	os_mutex_unlock(&ctdc->lock);
}

static void
constellation_tracked_device_connection_notify_position(struct t_constellation_tracked_device_connection *ctdc,
                                                        timepoint_ns frame_mono_ns,
                                                    const struct t_estimator_prior *prior,
                                                        const struct xrt_vec3 *position,
                                                        const struct xrt_vec3 *position_variance,
                                                        bool refresh_optical_anchor)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && prior_epoch_matches_locked(ctdc, prior) && ctdc->cb->push_observed_position) {
		ctdc->cb->push_observed_position(ctdc->xdev, frame_mono_ns, position, position_variance,
		                                  refresh_optical_anchor);
	}
	os_mutex_unlock(&ctdc->lock);
}

static void
constellation_tracked_device_connection_cache_pnp_pose_candidate(struct t_constellation_tracked_device_connection *ctdc,
                                                                timepoint_ns frame_mono_ns,
                                                    const struct t_estimator_prior *prior,
                                                                const struct xrt_pose *pose)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && prior_epoch_matches_locked(ctdc, prior) && ctdc->cb->cache_pnp_pose_candidate) {
		ctdc->cb->cache_pnp_pose_candidate(ctdc->xdev, frame_mono_ns, pose);
	}
	os_mutex_unlock(&ctdc->lock);
}

static void
constellation_tracked_device_connection_notify_world_reanchor(struct t_constellation_tracked_device_connection *ctdc,
                                                              timepoint_ns frame_mono_ns,
                                                              const struct xrt_pose *delta, const struct xrt_vec3 *new_raw_pivot,
                                                              timepoint_ns publication_ns, double gyro_dps, double speed_mps)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->notify_world_reanchor) {
		ctdc->cb->notify_world_reanchor(ctdc->xdev, frame_mono_ns, delta, new_raw_pivot,
                                               publication_ns, gyro_dps, speed_mps);
	}
	os_mutex_unlock(&ctdc->lock);
}

static void
constellation_tracked_device_connection_notify_leds(struct t_constellation_tracked_device_connection *ctdc,
                                                    timepoint_ns frame_mono_ns,
                                                    const struct t_estimator_prior *prior,
                                                    const struct xrt_pose *P_xrworld_cam,
                                                    const struct t_constellation_cam_calib *cam_calib,
                                                    const struct t_constellation_led_obs *leds,
                                                    size_t led_count)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && prior_epoch_matches_locked(ctdc, prior) && ctdc->cb->push_observed_leds) {
		ctdc->cb->push_observed_leds(ctdc->xdev, frame_mono_ns, P_xrworld_cam, cam_calib, leds, led_count);
	}
	os_mutex_unlock(&ctdc->lock);
}

/* Query the fusion's per-LED gate (zhat + 2x2 innovation covariance S) for one candidate blob<->LED
 * pairing. Returns false until the fusion is tracking, so partial fold is disabled at cold start. */
static bool
constellation_tracked_device_connection_predict_led_gate(struct t_constellation_tracked_device_connection *ctdc,
                                                         timepoint_ns when_ns,
                                                         const struct t_estimator_prior *prior,
                                                         const struct xrt_pose *P_xrworld_cam,
                                                         const struct t_constellation_cam_calib *cam_calib,
                                                         const struct xrt_vec3 *led_obj,
                                                         float out_zhat[2],
                                                         float out_S[4])
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && prior != NULL) {
		ret = ctdc->cb->predict_led_gate_from_prior &&
		      ctdc->cb->predict_led_gate_from_prior(prior, P_xrworld_cam, cam_calib, led_obj, out_zhat, out_S);
	} else if (!ctdc->disconnected && ctdc->cb->predict_led_gate) {
		ret = ctdc->cb->predict_led_gate(ctdc->xdev, when_ns, P_xrworld_cam, cam_calib, led_obj,
		                                  out_zhat, out_S);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static void
constellation_tracked_device_connection_notify_brightness_update(struct t_constellation_tracked_device_connection *ctdc,
                                                                 uint8_t average_brightness)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->push_brightness_update) {
		ctdc->cb->push_brightness_update(ctdc->xdev, average_brightness);
	}
	os_mutex_unlock(&ctdc->lock);
}

static bool
constellation_tracked_device_connection_get_led_model(struct t_constellation_tracked_device_connection *ctdc,
                                                      struct t_constellation_led_model *led_model)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->get_led_model) {
		ret = ctdc->cb->get_led_model(ctdc->xdev, led_model);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static bool
constellation_tracked_device_connection_get_pose_uncertainty(struct t_constellation_tracked_device_connection *ctdc,
                                                             double *position_std,
                                                             double *orientation_std,
                                                             double *yaw_std,
                                                             double *tilt_std)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->get_pose_uncertainty) {
		ret = ctdc->cb->get_pose_uncertainty(ctdc->xdev, position_std, orientation_std, yaw_std, tilt_std);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static bool
constellation_tracked_device_connection_get_gravity_tilt_reference(
    struct t_constellation_tracked_device_connection *ctdc,
    struct xrt_quat *out_gravity_corrected_q,
    double *out_excess_m_s2)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->get_gravity_tilt_reference) {
		ret = ctdc->cb->get_gravity_tilt_reference(ctdc->xdev, out_gravity_corrected_q, out_excess_m_s2);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static bool
constellation_tracked_device_connection_get_tracked_pose(struct t_constellation_tracked_device_connection *ctdc,
                                                         uint64_t timestamp_ns,
                                                         struct xrt_space_relation *xsr)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected) {
		struct xrt_device *xdev = ctdc->xdev;
		xrt_device_get_tracked_pose(xdev, XRT_INPUT_GENERIC_TRACKER_POSE, timestamp_ns, xsr);
		ret = true;
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

//! The fusion's RAW predicted prior (no body-lock/reach/re-entry), the honest estimate to gate + flip-cost
//! against. Optional callback; returns false if the device doesn't expose it (caller falls back to the
//! reported pose).
static bool
constellation_tracked_device_connection_get_predicted_pose(struct t_constellation_tracked_device_connection *ctdc,
                                                            uint64_t when_ns,
                                                            struct xrt_space_relation *xsr)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->get_predicted_pose) {
		ret = ctdc->cb->get_predicted_pose(ctdc->xdev, when_ns, xsr);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static bool
constellation_tracked_device_connection_get_last_optical_age_ms(struct t_constellation_tracked_device_connection *ctdc,
                                                                uint64_t when_ns,
                                                                double *age_ms)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->get_last_optical_age_ms) {
		ret = ctdc->cb->get_last_optical_age_ms(ctdc->xdev, when_ns, age_ms);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static void
constellation_tracker_receive_frame(struct xrt_frame_sink *sink, struct xrt_frame *xf)
{
	struct t_constellation_tracker *ct = container_of(sink, struct t_constellation_tracker, base);

	assert(xf->format == XRT_FORMAT_L8);

	// Tell the controllers about the frame so they can their timesync estimate
	os_mutex_lock(&ct->tracked_device_lock);
	for (int i = 0; i < ct->num_devices; i++) {
		constellation_tracked_device_connection_notify_frame(ct->devices[i].connection, xf->timestamp,
		                                                     xf->source_sequence);
	}
	os_mutex_unlock(&ct->tracked_device_lock);

	ct->last_frame_timestamp = xf->timestamp;
	xrt_sink_push_frame(ct->fast_q_sink, xf);
}

static void
constellation_tracker_node_break_apart(struct xrt_frame_node *node)
{
	DRV_TRACE_MARKER();
}

static void
mark_matching_blobs(struct t_constellation_tracker *ct,
                    struct xrt_pose *pose,
                    struct blobservation *bwobs,
                    struct t_constellation_led_model *led_model,
                    struct pose_metrics_blob_match_info *blob_match_info)
{
	/* First clear existing blob labels for this device */
	int i;
	for (i = 0; i < bwobs->num_blobs; i++) {
		struct blob *b = bwobs->blobs + i;
		uint32_t led_object_id = LED_OBJECT_ID(b->led_id);

		/* Skip blobs which already have an ID not belonging to this device */
		if (led_object_id != led_model->id) {
			continue;
		}

		if (b->led_id != LED_INVALID_ID) {
			b->prev_led_id = b->led_id;
		}
		b->led_id = LED_INVALID_ID;
	}


	/* Iterate the visible LEDs and mark matching blobs with this device ID and LED ID */
	for (i = 0; i < blob_match_info->num_visible_leds; i++) {
		struct pose_metrics_visible_led_info *led_info = blob_match_info->visible_leds + i;
		struct t_constellation_led *led = led_info->led;

		if (led_info->matched_blob != NULL) {
			struct blob *b = led_info->matched_blob;

			b->led_id = LED_MAKE_ID(led_model->id, led->id);
			CT_DEBUG(ct, "Marking LED %d/%d at %f,%f angle %f now %d (was %d)", led_model->id, led->id,
			         b->x, b->y, RAD_TO_DEG(acosf(led_info->facing_dot)), b->led_id, b->prev_led_id);
		} else {
			CT_DEBUG(ct, "No blob for device %d LED %d @ %f,%f size %f px angle %f", led_model->id, led->id,
			         led_info->pos_px.x, led_info->pos_px.y, 2 * led_info->led_radius_px,
			         RAD_TO_DEG(acosf(led_info->facing_dot)));
		}
	}
}

/* Undistorted normalized ray -> undistorted PIXEL via the real pinhole intrinsics, so the
 * fusion reprojects in physical pixels and its noise/gate are focal-independent. */
static struct xrt_vec2
blob_undistorted_px(const struct constellation_tracker_camera_state *cam, float blob_x, float blob_y)
{
	float nx = 0.f, ny = 0.f;
	t_camera_models_undistort(&cam->camera_model.calib, blob_x, blob_y, &nx, &ny);
	return (struct xrt_vec2){cam->camera_model.calib.fx * nx + cam->camera_model.calib.cx,
	                         cam->camera_model.calib.fy * ny + cam->camera_model.calib.cy};
}

/* Feed this view's matched LEDs to the controller fusion as per-LED reprojection observations.
 * Caller must have matched dev_state->blob_match_info to the folded pose. Deduplicated per view via
 * led_emit_view_mask. Frames: led_obj = P_device_model . P_YZ(led->pos) (model->device, OpenCV->OpenXR)
 * and the extrinsic is P_cam_world(CV) . P_YZ, so the filter reproduces the constellation projection. */
static void
emit_view_led_observations(struct tracking_sample_device_state *dev_state,
                           struct constellation_tracker_device *device,
                           struct constellation_tracker_camera_state *cam,
                           struct tracking_sample_frame *view,
                           int view_id,
                           timepoint_ns sample_ts)
{
	if (view_id < 0 || view_id >= 16 || (dev_state->led_emit_view_mask & (1u << view_id)) != 0) {
		return; // out of range, or already folded this view this sample
	}

	struct t_constellation_led_obs led_obs[MAX_OBJECT_LEDS];
	int n_led_obs = 0;
	for (int i = 0; i < dev_state->blob_match_info.num_visible_leds; i++) {
		struct pose_metrics_visible_led_info *visible_led = &dev_state->blob_match_info.visible_leds[i];
		if (visible_led->matched_blob == NULL) {
			continue;
		}
		led_obs[n_led_obs].obs_px =
		    blob_undistorted_px(cam, visible_led->matched_blob->x, visible_led->matched_blob->y);
		led_obs[n_led_obs].pos_var_px2 = visible_led->matched_blob->pos_var_px2;
		// led->pos is LED-MODEL frame; the fusion tracks the DEVICE pose, so map model->device after
		// the OpenCV->OpenXR YZ flip (mirrors the forward model flip(P_xrworld_device . P_device_model)).
		// Negating Y,Z is P_YZ_FLIP applied to a point (the explicit form is cheapest for a single point).
		struct xrt_vec3 led_flip = {visible_led->led->pos.x, -visible_led->led->pos.y,
		                            -visible_led->led->pos.z};
		math_pose_transform_point(&device->led_model.P_device_model, &led_flip,
		                          &led_obs[n_led_obs].led_obj);
		n_led_obs++;
	}
	if (n_led_obs == 0) {
		return;
	}

	struct xrt_pose P_xrworld_cam;
	math_pose_transform(&view->P_cam_world, &P_YZ_FLIP, &P_xrworld_cam);
	const struct t_constellation_cam_calib cam_calib = {cam->camera_model.calib.fx, cam->camera_model.calib.fy,
	                                                    cam->camera_model.calib.cx, cam->camera_model.calib.cy};
	constellation_tracked_device_connection_notify_leds(device->connection, sample_ts, dev_state->estimator_prior, &P_xrworld_cam,
	                                                    &cam_calib, led_obs, (size_t)n_led_obs);
	dev_state->led_emit_view_mask |= (1u << view_id);
}

static void
submit_device_pose(struct t_constellation_tracker *ct,
                   struct tracking_sample_device_state *dev_state,
                   struct constellation_tracking_sample *sample,
                   int view_id,
                   struct xrt_pose *P_cam_obj)
{
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct pose_metrics *score = &dev_state->score;
	if (!constellation_tracked_device_connection_prior_current(device->connection, dev_state->estimator_prior)) { return; }

	mark_matching_blobs(ct, P_cam_obj, view->bwobs, &device->led_model, &dev_state->blob_match_info);

	blobwatch_update_labels(cam->bw, view->bwobs, device->led_model.id);

	/* Telemetry: the accepted optical pose (outcome=1), camera-relative [p,q]. Called at most once
	 * per device per sample (additional contributing views go through
	 * association_fold_hypothesis_view), so this is always the lock (re)acquisition for the sample. */
	if (g2_telem_enabled()) {
		uint8_t dev_id = telem_device_id(device->connection->xdev);
		float pose7[7];
		telem_pack_pose(P_cam_obj, pose7);
		g2_telem_pose_attempt(dev_id, (uint8_t)view_id, (uint64_t)sample->timestamp,
		                      (uint8_t)score->visible_leds, (uint8_t)score->matched_blobs,
		                      (uint8_t)score->matched_blobs, (float)score->reprojection_error, pose7,
		                      /* outcome */ 1 /* accepted */);
		g2_telem_event(dev_id, (uint64_t)sample->timestamp, G2_TELEM_EV_LOCK_ACQUIRED, 0.0f);
	}

	math_pose_transform(&view->P_world_cam, P_cam_obj, &dev_state->final_pose);
	dev_state->found_device_pose = true;
	dev_state->found_pose_view_id = view_id;

	os_mutex_lock(&ct->tracked_device_lock);
	if (device->have_last_seen_pose == false || sample->timestamp > device->last_seen_pose_ts) {
		device->have_last_seen_pose = true;
		device->last_seen_pose_ts = sample->timestamp;
		device->last_seen_pose = dev_state->final_pose;
		device->last_matched_blobs = score->matched_blobs;
		device->last_matched_cam = view_id;
		device->last_matched_cam_pose = *P_cam_obj;

		/* Cache controller-in-head orientation for the next frame's head-anchored flip cue
		 * (Fix A): P_imu_obj = P_imu_cam . P_cam_obj. The TTL on the consumer side keeps a
		 * stale cache from biasing twin selection across dropouts. */
		struct xrt_pose P_imu_obj_cache;
		math_pose_transform(&cam->P_imu_cam, P_cam_obj, &P_imu_obj_cache);
		device->last_head_rel_quat = P_imu_obj_cache.orientation;
		device->last_head_rel_quat_ts = sample->timestamp;
		device->have_last_head_rel_quat = true;

		device->temporal_yaw_ref_quat = dev_state->final_pose.orientation;
		device->temporal_yaw_ref_ts = sample->timestamp;
		device->have_temporal_yaw_ref = true;
		device->temporal_yaw_ref_prior = dev_state->P_world_obj_prior.orientation;
		device->have_temporal_yaw_prior = true;

		/* Submit this pose observation to the fusion / real device. Flip back to OpenXR coords first, then
		 * apply model pose */
		struct xrt_pose P_xrworld_model;
		pose_flip_YZ(&dev_state->final_pose, &P_xrworld_model);

		// Apply device -> LED model pose from xsr = P_world_device + P_device_model = model pose
		struct xrt_pose P_xrworld_device;
		math_pose_transform(&P_xrworld_model, &device->led_model.P_model_device, &P_xrworld_device);

		// Average matched-blob brightness for the LED-intensity / brightness feedback. (The per-LED
		// fusion feed is emitted by emit_view_led_observations above, per view.)
		uint32_t average_brightness = 0;
		int matched_blobs = 0;
		for (int i = 0; i < dev_state->blob_match_info.num_visible_leds; i++) {
			struct pose_metrics_visible_led_info *visible_led = &dev_state->blob_match_info.visible_leds[i];
			if (visible_led->matched_blob) {
				average_brightness += visible_led->matched_blob->brightness;
				matched_blobs++;
			}
		}

		if (matched_blobs > 0) {
			average_brightness /= matched_blobs;
			constellation_tracked_device_connection_notify_brightness_update(device->connection,
			                                                                 average_brightness);
		}

		constellation_tracked_device_connection_notify_pose(device->connection, sample->timestamp, dev_state->estimator_prior,
		                                                    &P_xrworld_device);
	}
	os_mutex_unlock(&ct->tracked_device_lock);

	/* Fold this view's matched LEDs into the fusion after the accepted PnP pose has refreshed the ESKF's
	 * same-timestamp pose cache/bootstrap. Folding first made good lock frames look like zero-folds whenever
	 * the incoming prior was stale; the divergence re-anchor had no fresh PnP target yet. */
	emit_view_led_observations(dev_state, device, cam, view, view_id, sample->timestamp);
}

/* Pose-predicted LED label propagation: project the fusion's PREDICTED controller pose's LED
 * model into one view and assign each blob to the LED it lands on, back-face culled by the LED normals
 * and bounded by the fixed per-LED radius gate (all inside pose_metrics_match_pose_to_blobs).
 * This is the PROJECTED-LED-motion prior — the labels follow the predicted pose through head and
 * controller motion — NOT raw pixel velocity, which parallax and ego-motion make unreliable. Returns
	 * the count of blobs newly labelled to this device, so the caller can decide a view carries enough
	 * propagated IDs to solve. The label transfer
 * itself is mark_matching_blobs; this just wraps the project+label so the joint and single-cam paths
 * seed labels identically (one source of truth for "label a view from the predicted pose"). */
static int
device_propagate_labels_in_view(struct t_constellation_tracker *ct,
                                struct tracking_sample_device_state *dev_state,
                                struct constellation_tracking_sample *sample,
                                int view_id)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;

	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return 0;
	}

	struct xrt_pose P_cam_obj_prior;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);

	pose_metrics_match_pose_to_blobs(&P_cam_obj_prior, view->bwobs->blobs, view->bwobs->num_blobs,
	                                 &device->led_model, &cam->camera_model, &dev_state->blob_match_info);
	mark_matching_blobs(ct, &P_cam_obj_prior, view->bwobs, &device->led_model, &dev_state->blob_match_info);

	int n_labelled = 0;
	for (int i = 0; i < dev_state->blob_match_info.num_visible_leds; i++) {
		if (dev_state->blob_match_info.visible_leds[i].matched_blob != NULL) {
			n_labelled++;
		}
	}
	return n_labelled;
}

static void
association_fold_triangulated_position(struct t_constellation_tracker *ct,
                                       struct tracking_sample_device_state *dev_state,
                                       struct constellation_tracking_sample *sample,
                                       const struct multicam_tri_result *res);

static bool
association_fold_raw_epipolar_position(struct t_constellation_tracker *ct,
                                       struct tracking_sample_device_state *dev_state,
                                       struct constellation_tracking_sample *sample);

static bool
association_fold_partial_triangulated_position(
    struct t_constellation_tracker *ct,
    struct tracking_sample_device_state *dev_state,
    struct constellation_tracking_sample *sample,
    struct blob view_blobs[CONSTELLATION_MAX_CAMERAS][ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS],
    const int view_counts[CONSTELLATION_MAX_CAMERAS])
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct multicam_tri_view tri_views[CONSTELLATION_MAX_CAMERAS];
	int n_tri_views = 0;

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		if (view_counts[view_id] <= 0) {
			continue;
		}
		tri_views[n_tri_views].blobs = view_blobs[view_id];
		tri_views[n_tri_views].num_blobs = view_counts[view_id];
		tri_views[n_tri_views].calib = &ct->cam[view_id].camera_model;
		tri_views[n_tri_views].P_world_cam = sample->views[view_id].P_world_cam;
		n_tri_views++;
	}
	if (n_tri_views < 2) {
		return false;
	}

	struct multicam_tri_result res = {0};
	if (!multicam_triangulate_position(tri_views, n_tri_views, &device->led_model,
	                                   &dev_state->P_world_obj_prior.orientation,
	                                   dev_state->prior_yaw_sigma_rad, PARTIAL_TRI_MIN_LEDS, &res)) {
		return false;
	}
	if (res.position_std_m > PARTIAL_TRI_MAX_STD_M) {
		return false;
	}

	association_fold_triangulated_position(ct, dev_state, sample, &res);
	return true;
}

/* Partial-information fold with a covariance gate.
 *
 * When no pose hypothesis is reliable enough to lock, this folds the few LEDs that are confidently
 * matched to the prior, without committing a pose: for each prior-visible LED we
 * ask the fusion for its predicted image point + 2x2 innovation covariance S = H·P·Hᵀ + R
 * (predict_led_gate), then accept the blob whose Mahalanobis distance d²=rᵀS⁻¹r is smallest AND
 * ≤ χ²₂(0.99). Only in-gate LEDs are folded (the ESKF then grows covariance honestly on 1-3 LEDs).
 *
 * Guards (the contract):
	 *  - COLD START / no prior: predict_led_gate returns false until the fusion is tracking, AND we require
	 *    prior_tilt_trusted (= the fusion is tracking, gravity-anchored prior available). Untracked => no-op.
 *  - MISLABEL: the anisotropic S is the guard. A flipped/garbage correspondence reprojects far from
 *    zhat (in tilt especially — S is tight there), so d² blows past the gate and the LED is NOT folded.
 *    The per-LED χ² gate inside fold_led_observations is the second, consistent line of defence.
 *  - Each blob is assigned to at most one LED: LED-order greedy (LEDs scanned in index order, each claims
 *    its min-d² of the still-free in-gate blobs; blob_taken[] enforces one blob per LED and one LED per
 *    blob). Deterministic; harmless with 1-3 sparse LEDs (each fold is re-gated by the ESKF's own χ²).
 *
 * Does NOT report a pose (no submit_device_pose, no PnP "accept"); it only feeds the filter, so the
	 * device keeps reporting its covariance-grown prior while cold-search candidates handle genuine
	 * (re)acquire. Returns true iff at least one LED was gate-folded (diagnostic). A <4-LED frame is
	 * used productively here; the accept/flip decision stays with the anisotropic prior-cost paths (the
	 * cold-search accept is flip-ranked by the same prior split when a prior exists). */
static bool
association_fold_prior_leds(struct t_constellation_tracker *ct,
                            struct tracking_sample_device_state *dev_state,
                            struct constellation_tracking_sample *sample)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

		/* No reliable prior to gate with (cold start) -> no-op. The covariance gate
		 * needs a tracking filter (predict_led_gate returns false otherwise); prior_tilt_trusted is exactly
	 * "the fusion is tracking" (the gravity-anchored prior is available, even through a dropout). The
	 * anisotropic S handles tilt/yaw weighting itself, so no yaw-trust requirement here. */
	if (!dev_state->prior_tilt_trusted) {
		return false;
	}
	bool folded_any = false;
	struct pose_metrics_blob_match_info view_match_info[CONSTELLATION_MAX_CAMERAS];
	bool view_has_matches[CONSTELLATION_MAX_CAMERAS] = {false};
	struct blob view_blobs[CONSTELLATION_MAX_CAMERAS][ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS];
	int view_counts[CONSTELLATION_MAX_CAMERAS] = {0};

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;

		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}
		blobservation *bwobs = view->bwobs;

		struct xrt_pose P_cam_obj_prior;
		math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);

		/* Enumerate the prior-visible (front-facing, in-frame) LEDs for this view. We do NOT use the
		 * fixed-radius matched_blob it fills — the covariance gate below makes the real association. */
		pose_metrics_match_pose_to_blobs(&P_cam_obj_prior, bwobs->blobs, bwobs->num_blobs, &device->led_model,
		                                 &cam->camera_model, &dev_state->blob_match_info);

		/* The extrinsic + intrinsics the fusion gate must use (identical to emit_view_led_observations). */
		struct xrt_pose P_xrworld_cam;
		math_pose_transform(&view->P_cam_world, &P_YZ_FLIP, &P_xrworld_cam);
		const struct t_constellation_cam_calib cam_calib = {
		    cam->camera_model.calib.fx, cam->camera_model.calib.fy, cam->camera_model.calib.cx,
		    cam->camera_model.calib.cy};

		/* Precompute each blob's undistorted PIXEL position once (matches the gate's zhat units). */
		struct xrt_vec2 blob_px[MAX_BLOBS_PER_FRAME];
		bool blob_taken[MAX_BLOBS_PER_FRAME] = {false};
		const int num_blobs = bwobs->num_blobs < MAX_BLOBS_PER_FRAME ? bwobs->num_blobs : MAX_BLOBS_PER_FRAME;
		for (int b = 0; b < num_blobs; b++) {
			blob_px[b] = blob_undistorted_px(cam, bwobs->blobs[b].x, bwobs->blobs[b].y);
		}

		int n_matched = 0;
		for (int i = 0; i < dev_state->blob_match_info.num_visible_leds; i++) {
			struct pose_metrics_visible_led_info *visible_led = &dev_state->blob_match_info.visible_leds[i];
			visible_led->matched_blob = NULL; /* covariance gate decides; ignore the radius match */

			/* led_obj in the OpenXR object frame, exactly as emit_view_led_observations builds it. */
			struct xrt_vec3 led_flip = {visible_led->led->pos.x, -visible_led->led->pos.y,
			                            -visible_led->led->pos.z};
			struct xrt_vec3 led_obj;
			math_pose_transform_point(&device->led_model.P_device_model, &led_flip, &led_obj);

			float zhat[2], S[4];
			if (!constellation_tracked_device_connection_predict_led_gate(
			        device->connection, sample->timestamp, dev_state->estimator_prior, &P_xrworld_cam, &cam_calib, &led_obj,
			        zhat, S)) {
				/* Untracked or non-finite for this LED. Keep testing the remaining LEDs: edge-FOV
				 * sparse frames often have only a few useful LEDs, and one bad projection must not
				 * suppress the whole partial-information fold. */
				continue;
			}

			/* S = [[s0,s1],[s2,s3]]; S^-1 = 1/det [[s3,-s1],[-s2,s0]]. d² = rᵀ S⁻¹ r. */
			const double det = (double)S[0] * S[3] - (double)S[1] * S[2];
			if (!(det > 1e-9)) {
				continue; /* degenerate covariance -> skip this LED (no fold) */
			}
			const double inv00 = S[3] / det, inv01 = -S[1] / det, inv10 = -S[2] / det, inv11 = S[0] / det;

			int best_b = -1;
			double best_d2 = PARTIAL_FOLD_CHI2_2DOF;
			for (int b = 0; b < num_blobs; b++) {
				if (blob_taken[b]) {
					continue;
				}
				/* Skip blobs already labelled to ANOTHER device (unlabelled == LED_INVALID_ID is OK). */
				const uint16_t bid = bwobs->blobs[b].led_id;
				if (bid != LED_INVALID_ID && LED_OBJECT_ID(bid) != device->led_model.id) {
					continue;
				}
				const double rx = (double)blob_px[b].x - zhat[0];
				const double ry = (double)blob_px[b].y - zhat[1];
				const double d2 = rx * (inv00 * rx + inv01 * ry) + ry * (inv10 * rx + inv11 * ry);
				if (d2 < best_d2) {
					best_d2 = d2;
					best_b = b;
				}
			}
			if (best_b >= 0) {
				visible_led->matched_blob = &bwobs->blobs[best_b];
				blob_taken[best_b] = true;
				if (view_counts[view_id] < ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS) {
					struct blob labelled = bwobs->blobs[best_b];
					labelled.led_id = LED_MAKE_ID(device->led_model.id, visible_led->led->id);
					view_blobs[view_id][view_counts[view_id]++] = labelled;
				}
				n_matched++;
			}
		}

		/* Fold ONLY the covariance-gated LEDs (each re-gated by the fusion's own per-LED χ²). One or two
		 * gated LEDs are enough to keep the filter informed without committing a (flip-prone) pose. */
		if (n_matched > 0) {
			view_match_info[view_id] = dev_state->blob_match_info;
			view_has_matches[view_id] = true;
			folded_any = true;
			if (g2_telem_enabled()) {
					g2_telem_event(telem_device_id(device->connection->xdev), (uint64_t)sample->timestamp,
					               G2_TELEM_EV_PARTIAL_FOLD_COUNT, (float)n_matched);
			}
		}
	}

	const bool tri_folded =
	    association_fold_partial_triangulated_position(ct, dev_state, sample, view_blobs, view_counts);
	const bool raw_epipolar_folded =
	    tri_folded ? false : association_fold_raw_epipolar_position(ct, dev_state, sample);
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		if (!view_has_matches[view_id]) {
			continue;
		}
		dev_state->blob_match_info = view_match_info[view_id];
		emit_view_led_observations(dev_state, device, ct->cam + view_id, sample->views + view_id, view_id,
		                           sample->timestamp);
	}

	return folded_any || tri_folded || raw_epipolar_folded;
}

#define ASSOC_CLUTTER_NLL 0.2f
/* Retention evidence inside scoring (H2, results/h2-design-20260703): matching a
 * STATIC_CLUTTER blob (world-static >= STATIC_MAP_STATIC_DWELL_S outside every device's
 * exemption, static_map.c) as an LED pays clutter-vs-LED log-odds nats, ramping smoothly from
 * 0 at the class boundary (no knife-edge) to the per-blob price at the saturation dwell,
 * capped per hypothesis. Measured on the clutter benchmark: 0/1114 accepted true fits match
 * any STATIC blob vs 23.9% of wrong candidates (the t+41.519 lock-out fit: 6/7 blobs at
 * ~3.8 s dwell). Per-blob evidence within one hypothesis is correlated (same map region and
 * episode) and the exemption is imperfect for a still controller lost > the map gap, so the
 * cap keeps the worst case overridable by high-evidence fits - evidence, never a veto. */
#define ASSOC_RETENTION_STATIC_NLL 2.0f
#define ASSOC_RETENTION_DWELL_SAT_S 3.0f
#define ASSOC_RETENTION_TOTAL_MAX_NLL 6.0f
#define ASSOC_POS_PRIOR_WEIGHT 0.25f
#define ASSOC_VISUAL_LOCK_COST 10.0f
#define ASSOC_VISUAL_SINGLE_VIEW_LOW_LED_LOCK_COST 8.0f
#define ASSOC_VISUAL_LOCK_HIGH_EVIDENCE_COST 16.0f
#define ASSOC_VISUAL_HIGH_EVIDENCE_NLL 14.0f
#define ASSOC_VISUAL_RECOVERY_PRIOR_NLL 6.0f
#define ASSOC_VISUAL_RECOVERY_ACTION_NLL 2.0f
#define ASSOC_VISUAL_AMBIG_COST 20.0f
/* Cap on the (never-waived) gravity-tilt prior charge inside the recovery re-price: keeps a genuinely
 * tilt-poisoned prior (a previously committed flip) recoverable by overwhelming visual evidence — an
 * uncertainty-scaled cost, not a veto. */
#define ASSOC_VISUAL_RECOVERY_TILT_MAX_NLL ASSOC_VISUAL_AMBIG_COST
/* High-evidence tilt-trust envelope: the Huber quadratic zone of the tilt prior channel, i.e. the
 * candidate's tilt-vs-prior within PRIOR_GATE_SIGMA sigmas of the live (floored/ceilinged) tilt scale. */
#define ASSOC_HIGH_EVIDENCE_TILT_TRUST_NLL \
	((float)(FLIP_COST_WEIGHT * FLIP_COST_HUBER_KNEE_SIGMA * FLIP_COST_HUBER_KNEE_SIGMA))
#define ASSOC_ABSENT_WITH_BLOBS_COST 14.0f
#define ASSOC_MATCH_ALL_BLOBS_MAX 16
#define ASSOC_POSITION_ONLY_MIN_MATCHED 4
#define ASSOC_POSITION_ONLY_MAX_REPROJ_PER_LED 3.0f
#define ASSOC_POSITION_ONLY_ACTION_NLL 3.0f
#define ASSOC_TILT_CLAMP_MAX_REPROJ_PER_LED 3.0
#define ASSOC_TILT_CLAMP_MIN_DELTA_RAD DEG_TO_RAD(6.0)
#define ASSOC_GRAVITY_CLEAN_BAND_M_S2 1.0
#define ASSOC_LED_FOLD_ACTION_NLL 7.0f
/* Per-matched-blob reprojection NLL (Cauchy-robust, nats) for the SELECTION channel. The mean-px
 * reprojection term is ~1-nat scale while detection/cardinality terms are per-LED nats, so a
 * sub-pixel 8-LED fit could lose to a 6.5px 10-LED cross-fit (the hands-close identity swap).
 * Pricing each matched residual in nats makes fit quality dominate at the same per-LED scale. */
#define ASSOC_FIT_SIGMA2_PX2 1.0
#define ASSOC_FIT_CAUCHY_K2 4.0
#define ASSOC_LOCK_MATCH_EVIDENCE_NLL 0.35f
#define ASSOC_LOCK_MULTIVIEW_EVIDENCE_NLL 1.0f
#define ASSOC_LOCK_STRONG_EVIDENCE_NLL 1.0f
#define ASSOC_LOCK_MAX_EVIDENCE_NLL 5.0f
#define ASSOC_ORIENT_CONSENSUS_POS_GATE_M 0.06f
#define ASSOC_ORIENT_CONSENSUS_FREE_RAD DEG_TO_RAD(5.0)
#define ASSOC_ORIENT_CONSENSUS_SIGMA_RAD DEG_TO_RAD(10.0)
#define ASSOC_ORIENT_CONSENSUS_MAX_NLL 4.0f
#define ASSOC_TEMPORAL_YAW_TTL_NS HEAD_YAW_CUE_TTL_NS
#define ASSOC_TEMPORAL_YAW_SIGMA_RAD DEG_TO_RAD(45.0)
#define ASSOC_TEMPORAL_TILT_SIGMA_RAD DEG_TO_RAD(180.0)
#define ASSOC_TEMPORAL_MAX_NLL 3.0f
#define ASSOC_TEMPORAL_MAX_PROPAGATION_RAD DEG_TO_RAD(90.0)
#define ASSOC_TEMPORAL_TWIN_MIN_TILT_SEP_RAD DEG_TO_RAD(30.0)
#define ASSOC_YAW_BELIEF_TIE_NLL 0.5f
#define ASSOC_YAW_BELIEF_COMMIT_W 0.9f
#define ASSOC_YAW_BELIEF_MAX_FRAMES 4u
#define ASSOC_SINGLE_VIEW_PRIOR_MAX_NLL 6.0f
#define ASSOC_SINGLE_VIEW_PRIOR_MAX_DISP_M 0.8f
#define ASSOC_SINGLE_VIEW_PRIOR_DISP_MIN_NLL 1.0f
#define ASSOC_SINGLE_VIEW_MID_LED_MAX_REPROJ_PER_LED 1.5
#define ASSOC_SINGLE_VIEW_LOW_EVIDENCE_MAX_NLL 5.0f
#define ASSOC_SINGLE_VIEW_CLUTTER_MAX_BRIGHTNESS 30.0f
#define ASSOC_SINGLE_VIEW_CLUTTER_MAX_AREA 8.0f
#define ASSOC_SINGLE_VIEW_CLUTTER_MIN_VAR_PX2 3.0f
/* The two DN-denominated prongs of the single-view clutter veto above, re-denominated at the commanded
 * camera gain (B3 design §2/§6.3): clutter brightness scales with m = gain/16 exactly like LED brightness,
 * so an unscaled 30-DN bound silently DEACTIVATES at higher gain (clutter escapes above it); and the
 * variance prong tests the blobwatch g(b) inflation of the boundary-brightness blob, which moves with the
 * K gain law — it is re-derived as 3.0 * g_m(m*30) / g_16(30) with g(b) = 1 + (K/b)^2, so it keeps vetoing
 * the same physical blob population. At m = 1 both reduce exactly to the calibrated constants (old-capture
 * replays bit-identical). File-scope, set once at tracker create: the commanded gain is one operating point
 * per process (same pattern as the predictive-ROI env knobs below). */
static float assoc_single_view_clutter_max_brightness = ASSOC_SINGLE_VIEW_CLUTTER_MAX_BRIGHTNESS;
static float assoc_single_view_clutter_min_var_px2 = ASSOC_SINGLE_VIEW_CLUTTER_MIN_VAR_PX2;
#define ASSOC_STALE_PRIOR_RECOVERY_AGE_MS 120.0
#define ASSOC_STALE_PRIOR_RECOVERY_MIN_MATCHED 7
#define ASSOC_STALE_PRIOR_RECOVERY_MAX_UNMATCHED 2
#define ASSOC_STALE_PRIOR_RECOVERY_MAX_REPROJ_PER_LED 1.0f
#define ASSOC_STALE_PRIOR_RECOVERY_MAX_TOTAL_NLL ASSOC_VISUAL_LOCK_COST
#define ASSOC_COLD_PRIOR_ROI_MIN_MARGIN_PX 96.0f
#define ASSOC_COLD_PRIOR_ROI_MAX_MARGIN_PX 260.0f
#define ASSOC_COLD_PRIOR_ROI_ROT_LEVER_M 0.16f
#define ASSOC_POSITION_ONLY_BASE_STD_M 0.06f
#define ASSOC_POSITION_ONLY_PER_REPROJ_STD_M 0.02f
#define ASSOC_POSITION_ONLY_SINGLE_VIEW_INFLATE_M 0.04f
#define ASSOC_POSITION_ONLY_FEW_LED_INFLATE_M 0.03f
#define ASSOC_POSITION_ONLY_PARTIAL_INFLATE_M 0.05f
#define ASSOC_POSITION_ONLY_MIN_STD_M 0.04f
#define ASSOC_POSITION_ONLY_MAX_STD_M 0.20f
#define ASSOC_L2_MIN_MATCHED 6
#define ASSOC_L2_MIN_DISTINCT_VIEWS 2
#define ASSOC_L2_MAX_TILT_RAD DEG_TO_RAD(12.0)
#define ASSOC_L2_MAX_REPROJ_PER_LED_PX2 2.0
#define ASSOC_L2_MAX_TOTAL_NLL ASSOC_VISUAL_AMBIG_COST
#define ASSOC_L2_RECOVERY_ACTION_NLL 8.0f
#define ASSOC_L1_DISPUTE_SIGMA 3.0
#define ASSOC_L1_MIN_DISPUTE_LEDS 2
#define ASSOC_L1_REFINE_PER_CAM_LEDS 4
#define ASSOC_L1_REFINE_MAX_STD_M ASSOC_POSITION_ONLY_MIN_STD_M
#define ASSOC_L1_EPIPOLAR_MODEL_GATE_M 0.06f
#define ASSOC_L1_EPIPOLAR_REACH_SIGMA 3.0f
#define ASSOC_L1_EPIPOLAR_REACH_MIN_M 0.30f
#define ASSOC_L1_EPIPOLAR_REACH_MAX_M 1.35f
#define ASSOC_IDENTITY_STEAL_NEAR_M 0.12f
#define ASSOC_IDENTITY_STEAL_MARGIN_M 0.06f
#define ASSOC_IDENTITY_STEAL_SIGMA_M 0.04f
#define ASSOC_IDENTITY_STEAL_BASE_NLL 24.0f
#define ASSOC_IDENTITY_STEAL_VACATED_SCALE 0.35f
#define ASSOC_IDENTITY_STEAL_MAX_NLL 36.0f
/* Identity-ambiguity commitment margin (nats): a visual observation defers when an alternative
 * joint assignment that hands its cluster to a PARTNER device costs less than this much extra
 * (~e^3 = 20:1 odds). Sized above the measured knife-edge margin band (|margin| <~ 2 nats across
 * the 20260723 swap bifurcations, feed-epsilon flippable) and below the ~4-10 nat separation a
 * dense (m >= 8) cluster develops once chirality + gravity tilt discriminate the true owner. */
#define ASSOC_IDENTITY_DEFER_MARGIN_NLL 3.0f

enum association_observation_kind
{
	ASSOC_OBS_ABSENT = 0,
	ASSOC_OBS_POSE_LOCK = 1,
	ASSOC_OBS_POSITION_ONLY = 2,
	ASSOC_OBS_LED_FOLD = 3,
};

struct association_device_work
{
	struct association_pose_hypothesis hyps[ASSOCIATION_MAX_HYPOTHESES_PER_DEVICE];
	int count;
	bool has_blobs;
	bool folded_partial;
};

struct association_joint_choice
{
	const struct association_pose_hypothesis *chosen[CONSTELLATION_MAX_DEVICES];
	enum association_observation_kind kind[CONSTELLATION_MAX_DEVICES];
	float total_cost;
	int visual_count;
	bool valid;
};

struct association_blob_label_snapshot
{
	uint16_t led_id[MAX_BLOBS_PER_FRAME];
	uint16_t prev_led_id[MAX_BLOBS_PER_FRAME];
	int count;
};

static int
association_count_labelled_blobs(const blobservation *bwobs, uint16_t model_id)
{
	int n = 0;
	for (int i = 0; bwobs != NULL && i < bwobs->num_blobs; i++) {
		if (LED_OBJECT_ID(bwobs->blobs[i].led_id) == model_id) {
			n++;
		}
	}
	return n;
}

static void
association_save_labels(const blobservation *bwobs, struct association_blob_label_snapshot *snapshot)
{
	snapshot->count =
	    bwobs == NULL ? 0 : (bwobs->num_blobs < MAX_BLOBS_PER_FRAME ? bwobs->num_blobs : MAX_BLOBS_PER_FRAME);
	for (int i = 0; i < snapshot->count; i++) {
		snapshot->led_id[i] = bwobs->blobs[i].led_id;
		snapshot->prev_led_id[i] = bwobs->blobs[i].prev_led_id;
	}
}

static void
association_restore_labels(blobservation *bwobs, const struct association_blob_label_snapshot *snapshot)
{
	for (int i = 0; bwobs != NULL && i < snapshot->count && i < bwobs->num_blobs; i++) {
		bwobs->blobs[i].led_id = snapshot->led_id[i];
		bwobs->blobs[i].prev_led_id = snapshot->prev_led_id[i];
	}
}

static float
association_epipolar_reach_m(const struct tracking_sample_device_state *dev_state)
{
	float reach_m = ASSOC_L1_EPIPOLAR_REACH_SIGMA * m_vec3_len(dev_state->prior_pos_error);
	if (reach_m < ASSOC_L1_EPIPOLAR_REACH_MIN_M) {
		reach_m = ASSOC_L1_EPIPOLAR_REACH_MIN_M;
	}
	if (reach_m > ASSOC_L1_EPIPOLAR_REACH_MAX_M) {
		reach_m = ASSOC_L1_EPIPOLAR_REACH_MAX_M;
	}
	return reach_m;
}

/* The one identity-steal geometry shared by the hard pre-filter and the soft joint cost: how far a
 * candidate position is nearer the PARTNER's prior than its own, beyond the margin, given the partner
 * is close enough to contest. <= 0 (or non-finite/far partner) means no steal. */
static float
association_identity_steal_excess_m(float own_d, float other_d)
{
	if (!isfinite(other_d) || other_d > ASSOC_IDENTITY_STEAL_NEAR_M) {
		return 0.0f;
	}
	return own_d - other_d - ASSOC_IDENTITY_STEAL_MARGIN_M;
}

static bool
association_raw_epipolar_identity_safe(const struct constellation_tracking_sample *sample,
                                       const struct tracking_sample_device_state *dev_state,
                                       const struct multicam_tri_result *res)
{
	if (sample == NULL || dev_state == NULL || res == NULL || !dev_state->prior_tilt_trusted) {
		return false;
	}

	const float own_d = m_vec3_len(m_vec3_sub(res->position, dev_state->P_world_obj_prior.position));
	for (int i = 0; i < sample->n_devices; i++) {
		const struct tracking_sample_device_state *other = &sample->devices[i];
		if (other == dev_state || !other->prior_tilt_trusted) {
			continue;
		}
		const float other_d = m_vec3_len(m_vec3_sub(res->position, other->P_world_obj_prior.position));
		if (association_identity_steal_excess_m(own_d, other_d) > 0.0f) {
			return false;
		}
	}
	return true;
}

static bool
association_fold_raw_epipolar_position(struct t_constellation_tracker *ct,
                                       struct tracking_sample_device_state *dev_state,
                                       struct constellation_tracking_sample *sample)
{
	if (ct == NULL || dev_state == NULL || sample == NULL || !dev_state->prior_tilt_trusted) {
		return false;
	}

	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct blob view_blobs[CONSTELLATION_MAX_CAMERAS][MAX_BLOBS_PER_FRAME];
	struct multicam_tri_view views[CONSTELLATION_MAX_CAMERAS];
	int n_views = 0;
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs <= 0) {
			continue;
		}

		int n_blobs = 0;
		for (int b = 0; b < view->bwobs->num_blobs && n_blobs < MAX_BLOBS_PER_FRAME; b++) {
			const uint16_t led_id = view->bwobs->blobs[b].led_id;
			if (led_id != LED_INVALID_ID && LED_OBJECT_ID(led_id) != device->led_model.id) {
				continue;
			}
			view_blobs[n_views][n_blobs++] = view->bwobs->blobs[b];
		}
		if (n_blobs == 0) {
			continue;
		}

		views[n_views].blobs = view_blobs[n_views];
		views[n_views].num_blobs = n_blobs;
		views[n_views].calib = &ct->cam[view_id].camera_model;
		views[n_views].P_world_cam = view->P_world_cam;
		n_views++;
	}
	if (n_views < 2) {
		return false;
	}

	const float reach_m = association_epipolar_reach_m(dev_state);

	struct multicam_tri_result res = {0};
	if (!multicam_triangulate_epipolar_position(views, n_views, &device->led_model,
	                                            &dev_state->P_world_obj_prior,
	                                            dev_state->prior_yaw_sigma_rad, reach_m,
	                                            ASSOC_L1_EPIPOLAR_MODEL_GATE_M,
	                                            PARTIAL_TRI_MIN_LEDS, &res)) {
		return false;
	}
	if (res.num_views < 2 || res.num_leds < PARTIAL_TRI_MIN_LEDS ||
	    res.position_std_m > PARTIAL_TRI_MAX_STD_M ||
	    !association_raw_epipolar_identity_safe(sample, dev_state, &res)) {
		return false;
	}

	association_fold_triangulated_position(ct, dev_state, sample, &res);
	if (g2_telem_enabled()) {
		const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;
		g2_telem_event(telem_device_id(xdev), sample->timestamp, G2_TELEM_EV_ASSOC_RAW_EPIPOLAR_POSITION,
		               res.position_std_m);
	}
	return true;
}

static double
association_quat_angle(const struct xrt_quat *a, const struct xrt_quat *b)
{
	double d = fabs((double)a->x * b->x + (double)a->y * b->y + (double)a->z * b->z + (double)a->w * b->w);
	if (d > 1.0) {
		d = 1.0;
	}
	return 2.0 * acos(d);
}

static struct xrt_quat
association_propagate_temporal_ref(const struct xrt_quat *ref_quat,
                                   const struct xrt_quat *ref_prior,
                                   const struct xrt_quat *prior_now)
{
	struct xrt_quat prior_ref_inv;
	math_quat_invert(ref_prior, &prior_ref_inv);

	struct xrt_quat prior_delta;
	math_quat_rotate(prior_now, &prior_ref_inv, &prior_delta);
	math_quat_normalize(&prior_delta);

	if (association_quat_angle(ref_prior, prior_now) > ASSOC_TEMPORAL_MAX_PROPAGATION_RAD) {
		return *ref_quat;
	}

	struct xrt_quat out;
	math_quat_rotate(&prior_delta, ref_quat, &out);
	math_quat_normalize(&out);
	return out;
}

static double
association_capped_yaw_continuity_nll(const struct xrt_quat *candidate, const struct xrt_quat *reference)
{
	const struct xrt_vec3 world_up = {0.f, 1.f, 0.f};
	const double raw =
	    pose_metrics_prior_orient_cost(candidate, reference, &world_up, ASSOC_TEMPORAL_TILT_SIGMA_RAD,
	                                   ASSOC_TEMPORAL_YAW_SIGMA_RAD, FLIP_COST_HUBER_KNEE_SIGMA,
	                                   FLIP_COST_WEIGHT);
	return raw > (double)ASSOC_TEMPORAL_MAX_NLL ? (double)ASSOC_TEMPORAL_MAX_NLL : raw;
}

static float
association_temporal_yaw_nll(const struct constellation_tracker_device *device,
                             const struct tracking_sample_device_state *dev_state,
                             uint64_t timestamp_ns,
                             const struct xrt_pose *pose_world)
{
	if (device == NULL || dev_state == NULL || pose_world == NULL || !device->have_temporal_yaw_ref) {
		return 0.0f;
	}
	if (timestamp_ns < device->temporal_yaw_ref_ts ||
	    timestamp_ns - device->temporal_yaw_ref_ts >= (uint64_t)ASSOC_TEMPORAL_YAW_TTL_NS) {
		return 0.0f;
	}

	struct xrt_quat ref = device->temporal_yaw_ref_quat;
	if (device->have_temporal_yaw_prior) {
		ref = association_propagate_temporal_ref(&device->temporal_yaw_ref_quat,
		                                         &device->temporal_yaw_ref_prior,
		                                         &dev_state->P_world_obj_prior.orientation);
	}

	return (float)association_capped_yaw_continuity_nll(&pose_world->orientation, &ref);
}

static bool
association_pose_duplicate(const struct association_pose_hypothesis *a,
                           const struct association_pose_hypothesis *b)
{
	struct xrt_vec3 dp = m_vec3_sub(a->pose_world.position, b->pose_world.position);
	return m_vec3_len(dp) < 0.02 && association_quat_angle(&a->pose_world.orientation, &b->pose_world.orientation) <
	                                  DEG_TO_RAD(2.0);
}

static bool
association_lock_eligible(const struct association_pose_hypothesis *hyp);
static double
association_reprojection_per_match(const struct association_pose_hypothesis *hyp);

static float
association_orientation_consensus_unit_nll(double angle_rad)
{
	const double excess = angle_rad - ASSOC_ORIENT_CONSENSUS_FREE_RAD;
	if (excess <= 0.0) {
		return 0.0f;
	}
	const double x = excess / ASSOC_ORIENT_CONSENSUS_SIGMA_RAD;
	const double nll = x * x;
	return (float)(nll > ASSOC_ORIENT_CONSENSUS_MAX_NLL ? ASSOC_ORIENT_CONSENSUS_MAX_NLL : nll);
}

static bool
association_same_position_consensus_neighbour(const struct association_pose_hypothesis *a,
                                              const struct association_pose_hypothesis *b)
{
	if (a == b || b == NULL || b->matched_count < 4 || !POSE_HAS_FLAGS(&b->score, POSE_MATCH_POSITION)) {
		return false;
	}
	if (a->source == b->source && a->primary_view_id == b->primary_view_id) {
		return false;
	}
	struct xrt_vec3 dp = m_vec3_sub(a->pose_world.position, b->pose_world.position);
	return m_vec3_len(dp) <= ASSOC_ORIENT_CONSENSUS_POS_GATE_M;
}

static void
association_apply_orientation_consensus(struct association_device_work *work)
{
	for (int i = 0; work != NULL && i < work->count; i++) {
		struct association_pose_hypothesis *hyp = &work->hyps[i];
		hyp->cost.orientation_consensus_nll = 0.0f;
		if (!association_lock_eligible(hyp) || !POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_POSITION)) {
			continue;
		}

		double weighted_nll = 0.0;
		double total_weight = 0.0;
		int support = 0;
		for (int j = 0; j < work->count; j++) {
			const struct association_pose_hypothesis *other = &work->hyps[j];
			if (!association_same_position_consensus_neighbour(hyp, other)) {
				continue;
			}
			const double angle = association_quat_angle(&hyp->pose_world.orientation, &other->pose_world.orientation);
			const double weight = other->matched_count > 0 ? (double)other->matched_count : 1.0;
			weighted_nll += weight * association_orientation_consensus_unit_nll(angle);
			total_weight += weight;
			support++;
		}

		if (support >= 2 && total_weight > 0.0) {
			const double avg = weighted_nll / total_weight;
			hyp->cost.orientation_consensus_nll =
			    (float)(avg > ASSOC_ORIENT_CONSENSUS_MAX_NLL ? ASSOC_ORIENT_CONSENSUS_MAX_NLL : avg);
		}

	}
}

static int
association_distinct_view_count(const struct association_pose_hypothesis *hyp)
{
	uint16_t mask = 0;
	for (uint8_t i = 0; hyp != NULL && i < hyp->matched_count; i++) {
		const int view_id = hyp->matched_blobs[i].view_id;
		if (view_id >= 0 && view_id < 16) {
			mask |= (uint16_t)(1u << view_id);
		}
	}

	int count = 0;
	for (; mask != 0; mask = (uint16_t)(mask & (mask - 1))) {
		count++;
	}
	return count;
}

static float
association_axis_nll(float value, float sigma)
{
	if (!(sigma > 0.0f) || !isfinite(sigma)) {
		return 0.0f;
	}
	const float s = value / sigma;
	return 0.5f * s * s;
}

static float
association_position_prior_nll(const struct pose_metrics *score,
                               const struct tracking_sample_device_state *dev_state)
{
	if (!POSE_HAS_FLAGS(score, POSE_HAD_PRIOR)) {
		return 0.0f;
	}
	return ASSOC_POS_PRIOR_WEIGHT *
	       (association_axis_nll((float)score->pos_error.x, dev_state->prior_pos_error.x) +
	        association_axis_nll((float)score->pos_error.y, dev_state->prior_pos_error.y) +
	        association_axis_nll((float)score->pos_error.z, dev_state->prior_pos_error.z));
}

static float
association_orientation_prior_nll(const struct tracking_sample_device_state *dev_state,
                                  const struct xrt_pose *candidate,
                                  const struct xrt_pose *prior,
                                  const struct xrt_vec3 *up)
{
	if (!dev_state->prior_tilt_trusted) {
		return 0.0f;
	}
	return (float)pose_metrics_prior_orient_cost(&candidate->orientation, &prior->orientation, up,
	                                            dev_state->prior_tilt_sigma_rad, dev_state->prior_yaw_sigma_rad,
	                                            FLIP_COST_HUBER_KNEE_SIGMA, FLIP_COST_WEIGHT);
}

static float
association_head_anchor_nll(const struct constellation_tracker_device *device,
                            const struct constellation_tracker_camera_state *cam,
                            const struct tracking_sample_frame *view,
                            const struct constellation_tracking_sample *sample,
                            const struct xrt_pose *P_cam_obj)
{
	if (!device->have_last_head_rel_quat || sample->timestamp < device->last_head_rel_quat_ts ||
	    sample->timestamp - device->last_head_rel_quat_ts >= (uint64_t)HEAD_YAW_CUE_TTL_NS) {
		return 0.0f;
	}

	struct xrt_pose P_imu_obj;
	math_pose_transform(&cam->P_imu_cam, P_cam_obj, &P_imu_obj);

	struct xrt_vec3 imu_gravity;
	math_quat_rotate_vec3(&cam->P_imu_cam.orientation, &view->cam_gravity_vector, &imu_gravity);

	return (float)pose_metrics_prior_orient_cost(&P_imu_obj.orientation, &device->last_head_rel_quat,
	                                            &imu_gravity, HEAD_YAW_CUE_TILT_SIGMA,
	                                            HEAD_YAW_CUE_YAW_SIGMA, FLIP_COST_HUBER_KNEE_SIGMA,
	                                            FLIP_COST_WEIGHT);
}

/* Physical footprint of a controller's LED ring, for cross-device occlusion reasoning. */
#define ASSOC_PARTNER_BODY_RADIUS_M 0.09
#define ASSOC_PARTNER_SIGMA_FLOOR_M 0.01

/* Cross-device occlusion credit: the per-LED detection model assumes independent misses, but at
 * hands-close range the partner controller physically occludes/absorbs this device's LEDs. A
 * visible-but-unmatched LED under the partner's prior footprint is expected to be missing and
 * must not count as evidence against the hypothesis (without this, the honest high-visibility
 * candidate loses to a sloppy cross-fit and the devices can swap identities).
 *
 * The excuse is weighted by the PROBABILITY the partner actually covers the LED under its prior
 * (body disc of radius r_b, partner centre ~ isotropic Gaussian sigma_p in the image):
 *   p_occl = (1 - exp(-r_b^2 / (2 sigma_p^2))) * exp(-max(0, d - r_b)^2 / (2 sigma_p^2))
 * — the Gaussian mass within one body radius, decayed beyond the disc edge. A tightly-tracked
 * partner sitting on the LED excuses it fully (p_occl -> 1); a coasting partner with a wide
 * prior excuses almost nothing anywhere (its occupancy is spread over the whole uncertainty
 * area). The legacy hard disc of radius r_b + sigma at FULL strength did the inverse — the more
 * lost the partner, the larger the fully-excused region (up to ~14 nats/view on the felt-clutter
 * capture), letting a dim-clutter fit 1 m off the true track wipe out its own miss evidence
 * (proven frame-exact at 20260612-153651 band-2 t+41.519; results/h6-rmodel-20260612). */
static double
association_partner_occlusion_credit(const struct constellation_tracking_sample *sample,
                                     const struct tracking_sample_device_state *dev_state,
                                     const struct tracking_sample_frame *view,
                                     const struct constellation_tracker_camera_state *cam,
                                     const struct pose_metrics_blob_match_info *match_info)
{
	double centers_px[CONSTELLATION_MAX_DEVICES][2];
	double body_r_px[CONSTELLATION_MAX_DEVICES];
	double sigma_px[CONSTELLATION_MAX_DEVICES];
	int n_discs = 0;
	for (int s = 0; s < sample->n_devices; s++) {
		const struct tracking_sample_device_state *other = sample->devices + s;
		if (other == dev_state) {
			continue;
		}
		struct xrt_pose P_cam_other;
		math_pose_transform((struct xrt_pose *)&view->P_cam_world,
		                    (struct xrt_pose *)&other->P_world_obj_prior, &P_cam_other);
		if (!(P_cam_other.position.z > 0.05f)) {
			continue;
		}
		const float sx = other->prior_pos_error.x, sy = other->prior_pos_error.y;
		const float sz = other->prior_pos_error.z;
		float sigma = sx > sy ? sx : sy;
		sigma = sigma > sz ? sigma : sz;
		if (!isfinite(sigma) || sigma < 0.0f) {
			continue;
		}
		if (sigma < ASSOC_PARTNER_SIGMA_FLOOR_M) {
			sigma = ASSOC_PARTNER_SIGMA_FLOOR_M;
		}
		float u = 0.0f, v = 0.0f;
		if (!t_camera_models_project(&cam->camera_model.calib, P_cam_other.position.x,
		                             P_cam_other.position.y, P_cam_other.position.z, &u, &v)) {
			continue;
		}
		const double px_per_m = (double)cam->camera_model.calib.fx / (double)P_cam_other.position.z;
		centers_px[n_discs][0] = (double)u;
		centers_px[n_discs][1] = (double)v;
		body_r_px[n_discs] = ASSOC_PARTNER_BODY_RADIUS_M * px_per_m;
		sigma_px[n_discs] = (double)sigma * px_per_m;
		n_discs++;
	}
	if (n_discs == 0) {
		return 0.0;
	}

	double credit = 0.0;
	for (int i = 0; i < match_info->num_visible_leds; i++) {
		const struct pose_metrics_visible_led_info *led = &match_info->visible_leds[i];
		if (led->matched_blob != NULL) {
			continue;
		}
		double p_occl = 0.0;
		for (int d = 0; d < n_discs; d++) {
			const double dx = led->pos_px.x - centers_px[d][0];
			const double dy = led->pos_px.y - centers_px[d][1];
			const double dist = sqrt(dx * dx + dy * dy);
			const double two_var = 2.0 * sigma_px[d] * sigma_px[d];
			const double mass = 1.0 - exp(-(body_r_px[d] * body_r_px[d]) / two_var);
			const double over = dist > body_r_px[d] ? dist - body_r_px[d] : 0.0;
			const double p = mass * exp(-(over * over) / two_var);
			if (p > p_occl) {
				p_occl = p;
			}
		}
		/* Discount the miss cost this LED was ACTUALLY charged, which the merge model may already
		 * have reduced — crediting the undiscounted -log(1-p) here forgives a merged-and-occluded
		 * LED twice and spills the excess onto the matched LEDs' terms. */
		credit += p_occl * led->miss_nll;
	}
	return credit < match_info->data_nll_detection ? credit : match_info->data_nll_detection;
}

static float
association_prior_pose_nll(const struct constellation_tracker_device *device,
                           const struct constellation_tracker_camera_state *cam,
                           const struct tracking_sample_device_state *dev_state,
                           const struct tracking_sample_frame *view,
                           const struct constellation_tracking_sample *sample,
                           const struct xrt_pose *P_cam_obj)
{
	struct xrt_pose P_cam_obj_prior;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);
	struct xrt_vec3 dp = m_vec3_sub(P_cam_obj->position, P_cam_obj_prior.position);
	return ASSOC_POS_PRIOR_WEIGHT *
	           (association_axis_nll(dp.x, dev_state->prior_pos_error.x) +
	            association_axis_nll(dp.y, dev_state->prior_pos_error.y) +
	            association_axis_nll(dp.z, dev_state->prior_pos_error.z)) +
	       association_orientation_prior_nll(dev_state, P_cam_obj, &P_cam_obj_prior, &view->cam_gravity_vector) +
	       association_head_anchor_nll(device, cam, view, sample, P_cam_obj);
}

/* @p pose_cam_override, when non-NULL, is the pose expressed in THIS view's camera frame (the hypothesis
 * stores pose_cam in its PRIMARY view's frame). A committed multi-view pose is applied and folded into every
 * contributing view (association_fold_hypothesis_view), so each such view emits its own outcome=1 record with
 * the pose re-expressed in its frame; NULL keeps the primary-view pose_cam. */
static void
association_emit_candidate(const struct constellation_tracker_device *device,
                           const struct tracking_sample_device_state *dev_state,
                           const struct tracking_sample_frame *view,
                           int view_id,
                           uint64_t timestamp_ns,
                           const struct association_pose_hypothesis *hyp,
                           bool selected,
                           uint8_t outcome,
                           const struct xrt_pose *pose_cam_override)
{
	if (hyp == NULL || view == NULL) {
		return;
	}
	const struct xrt_pose *pose_cam = pose_cam_override != NULL ? pose_cam_override : &hyp->pose_cam;

	const float prior_pos_err[3] = {
	    (float)hyp->score.pos_error.x,
	    (float)hyp->score.pos_error.y,
	    (float)hyp->score.pos_error.z,
	};
	const float prior_rot_err[3] = {
	    (float)hyp->score.orient_error.x,
	    (float)hyp->score.orient_error.y,
	    (float)hyp->score.orient_error.z,
	};
	float pose[7];
	telem_pack_pose(pose_cam, pose);
	const struct xrt_device *xdev = device != NULL && device->connection != NULL ? device->connection->xdev : NULL;
	const float prior_cost =
	    hyp->cost.position_prior_nll + hyp->cost.orientation_prior_nll + hyp->cost.head_anchor_nll;
	g2_telem_candidate(telem_device_id(xdev), (uint8_t)view_id, timestamp_ns, hyp->source,
	                   (hyp->flags & ASSOC_HYP_IS_TWIN) ? 1 : 0, selected ? 1 : 0,
	                   (hyp->flags & ASSOC_HYP_HAS_TWIN) ? 1 : 0, outcome, hyp->score.match_flags,
	                   (uint8_t)hyp->visible_count, (uint8_t)hyp->matched_count,
	                   (uint8_t)hyp->unmatched_count, (uint8_t)hyp->matched_count,
	                   (float)hyp->score.reprojection_error, prior_cost, hyp->cost.total_nll,
	                   dev_state->prior_tilt_trusted ? 1 : 0, dev_state->prior_yaw_sigma_rad,
	                   hyp->tilt_error_rad, hyp->yaw_error_rad, hyp->tilt_prior_nll, prior_pos_err,
	                   prior_rot_err, hyp->blob_var_mean_px2, hyp->blob_brightness_mean,
	                   hyp->blob_area_mean, pose);
}

static int
association_blob_index(const blobservation *bwobs, const struct blob *blob)
{
	if (bwobs == NULL || blob == NULL) {
		return -1;
	}
	for (int i = 0; i < bwobs->num_blobs; i++) {
		if (&bwobs->blobs[i] == blob) {
			return i;
		}
	}
	return -1;
}

static int
association_fill_blob_refs(struct association_pose_hypothesis *hyp,
                           const struct pose_metrics_blob_match_info *match_info,
                           const blobservation *bwobs,
                           int view_id)
{
	int added = 0;
	for (int i = 0; i < match_info->num_visible_leds; i++) {
		const struct pose_metrics_visible_led_info *visible = &match_info->visible_leds[i];
		if (visible->matched_blob == NULL || visible->led == NULL) {
			continue;
		}
		const int blob_idx = association_blob_index(bwobs, visible->matched_blob);
		if (blob_idx < 0) {
			continue;
		}
		const uint8_t before = hyp->matched_count;
		association_hypothesis_add_blob(hyp, (int16_t)view_id, (int16_t)blob_idx, (int16_t)visible->led->id);
		added += hyp->matched_count > before ? 1 : 0;
	}
	return added;
}

static void
association_update_blob_quality(struct association_pose_hypothesis *hyp, struct constellation_tracking_sample *sample)
{
	float var_sum = 0.0f;
	float brightness_sum = 0.0f;
	float area_sum = 0.0f;
	uint8_t count = 0;
	for (uint8_t i = 0; hyp != NULL && sample != NULL && i < hyp->matched_count; i++) {
		const struct association_blob_ref *ref = &hyp->matched_blobs[i];
		if (!association_blob_ref_is_valid(ref) || ref->view_id < 0 || ref->view_id >= sample->n_views) {
			continue;
		}
		const blobservation *bwobs = sample->views[ref->view_id].bwobs;
		if (bwobs == NULL || ref->blob_id < 0 || ref->blob_id >= bwobs->num_blobs) {
			continue;
		}
		const struct blob *b = &bwobs->blobs[ref->blob_id];
		var_sum += b->pos_var_px2;
		brightness_sum += (float)b->brightness;
		area_sum += (float)b->area;
		count++;
	}
	hyp->blob_quality_count = count;
	if (count == 0) {
		hyp->blob_var_mean_px2 = 0.0f;
		hyp->blob_brightness_mean = 0.0f;
		hyp->blob_area_mean = 0.0f;
		return;
	}
	const float inv_n = 1.0f / (float)count;
	hyp->blob_var_mean_px2 = var_sum * inv_n;
	hyp->blob_brightness_mean = brightness_sum * inv_n;
	hyp->blob_area_mean = area_sum * inv_n;
}

static struct t_constellation_led *
association_find_led(struct t_constellation_led_model *led_model, int led_id)
{
	for (uint8_t i = 0; i < led_model->num_leds; i++) {
		if (led_model->leds[i].id == led_id) {
			return &led_model->leds[i];
		}
	}
	return NULL;
}

static int
association_apply_hypothesis_matches(struct tracking_sample_device_state *dev_state,
                                     struct constellation_tracker_device *device,
                                     struct tracking_sample_frame *view,
                                     const struct association_pose_hypothesis *hyp,
                                     int view_id)
{
	memset(&dev_state->blob_match_info, 0, sizeof(dev_state->blob_match_info));
	dev_state->blob_match_info.all_led_ids_matched = true;
	dev_state->blob_match_info.matched_blobs = hyp->matched_count;
	dev_state->blob_match_info.unmatched_blobs = hyp->unmatched_count;
	dev_state->blob_match_info.reprojection_error = hyp->cost.reprojection_nll * (double)hyp->matched_count;

	int written = 0;
	for (uint8_t i = 0; i < hyp->matched_count && written < MAX_OBJECT_LEDS; i++) {
		const struct association_blob_ref *ref = &hyp->matched_blobs[i];
		if (!association_blob_ref_is_valid(ref) || ref->view_id != view_id || ref->blob_id < 0 ||
		    ref->blob_id >= view->bwobs->num_blobs) {
			continue;
		}
		struct t_constellation_led *led = association_find_led(&device->led_model, hyp->matched_led_ids[i]);
		if (led == NULL) {
			continue;
		}
		struct pose_metrics_visible_led_info *dst = &dev_state->blob_match_info.visible_leds[written++];
		dst->led = led;
		dst->matched_blob = &view->bwobs->blobs[ref->blob_id];
		dst->pos_px = (struct xrt_vec2){dst->matched_blob->x, dst->matched_blob->y};
	}
	dev_state->blob_match_info.num_visible_leds = written;
	dev_state->blob_match_info.matched_blobs = written;
	return written;
}

static bool
association_hypothesis_has_view(const struct association_pose_hypothesis *hyp, int view_id)
{
	for (uint8_t i = 0; hyp != NULL && i < hyp->matched_count; i++) {
		if (hyp->matched_blobs[i].view_id == view_id) {
			return true;
		}
	}
	return false;
}

static int
association_fold_hypothesis_view(struct t_constellation_tracker *ct,
                                 struct tracking_sample_device_state *dev_state,
                                 struct constellation_tracking_sample *sample,
                                 const struct association_pose_hypothesis *hyp,
                                 int view_id,
                                 bool update_labels)
{
	if (hyp == NULL || view_id < 0 || view_id >= sample->n_views ||
	    !association_hypothesis_has_view(hyp, view_id)) {
		return 0;
	}
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return 0;
	}
	const int selected_matches = association_apply_hypothesis_matches(dev_state, device, view, hyp, view_id);
	if (selected_matches <= 0) {
		return 0;
	}
	if (update_labels) {
		struct xrt_pose P_cam_obj;
		math_pose_transform(&view->P_cam_world, &hyp->pose_world, &P_cam_obj);
		mark_matching_blobs(ct, &P_cam_obj, view->bwobs, &device->led_model, &dev_state->blob_match_info);
		blobwatch_update_labels(cam->bw, view->bwobs, device->led_model.id);
	}
	emit_view_led_observations(dev_state, device, cam, view, view_id, sample->timestamp);
	return selected_matches;
}

static bool
association_refine_multiview_pose(struct t_constellation_tracker *ct,
                                  struct tracking_sample_device_state *dev_state,
                                  struct constellation_tracking_sample *sample,
                                  const struct association_pose_hypothesis *hyp,
                                  struct xrt_pose *out_primary_cam_pose)
{
	if (hyp == NULL || out_primary_cam_pose == NULL || association_distinct_view_count(hyp) < 2) {
		return false;
	}

	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct blob view_blobs[JOINT_PNP_MAX_VIEWS][ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS];
	struct joint_pnp_view jviews[JOINT_PNP_MAX_VIEWS];
	int view_ids[JOINT_PNP_MAX_VIEWS];
	int view_counts[JOINT_PNP_MAX_VIEWS] = {0};
	int n_jviews = 0;

	for (uint8_t i = 0; i < hyp->matched_count; i++) {
		const struct association_blob_ref *ref = &hyp->matched_blobs[i];
		if (!association_blob_ref_is_valid(ref) || ref->view_id < 0 || ref->view_id >= sample->n_views ||
		    hyp->matched_led_ids[i] < 0 || hyp->matched_led_ids[i] >= device->led_model.num_leds) {
			continue;
		}
		struct tracking_sample_frame *view = sample->views + ref->view_id;
		if (view->bwobs == NULL || ref->blob_id < 0 || ref->blob_id >= view->bwobs->num_blobs) {
			continue;
		}

		int slot = -1;
		for (int j = 0; j < n_jviews; j++) {
			if (view_ids[j] == ref->view_id) {
				slot = j;
				break;
			}
		}
		if (slot < 0) {
			if (n_jviews >= JOINT_PNP_MAX_VIEWS) {
				continue;
			}
			slot = n_jviews++;
			view_ids[slot] = ref->view_id;
		}
		if (view_counts[slot] >= ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS) {
			continue;
		}
		struct blob labelled_blob = view->bwobs->blobs[ref->blob_id];
		labelled_blob.led_id = LED_MAKE_ID(device->led_model.id, hyp->matched_led_ids[i]);
		view_blobs[slot][view_counts[slot]++] = labelled_blob;
	}

	int contributing_views = 0;
	for (int j = 0; j < n_jviews; j++) {
		if (view_counts[j] == 0) {
			continue;
		}
		struct constellation_tracker_camera_state *cam = ct->cam + view_ids[j];
		jviews[contributing_views].blobs = view_blobs[j];
		jviews[contributing_views].num_blobs = view_counts[j];
		jviews[contributing_views].calib = &cam->camera_model;
		jviews[contributing_views].P_imu_cam = cam->P_imu_cam;
		contributing_views++;
	}
	if (contributing_views < 2) {
		return false;
	}

	struct xrt_pose P_imu_obj = hyp->pose_imu;
	int num_rays = 0;
	int num_inliers = 0;
	if (!joint_pnp_solve(&P_imu_obj, jviews, contributing_views, &device->led_model, &num_rays, &num_inliers)) {
		return false;
	}

	struct xrt_pose P_cam_imu;
	math_pose_invert(&ct->cam[hyp->primary_view_id].P_imu_cam, &P_cam_imu);
	math_pose_transform(&P_cam_imu, &P_imu_obj, out_primary_cam_pose);
	return true;
}

/* Selection ranks reprojection through TWO deliberate components: fit_quality_nll (per-blob Cauchy nats,
 * saturates ~2·log(r) on gross outliers) plus total_nll's mean-px term (1 px ≡ 1 nat, non-saturating, keeps
 * ordering sloppy fits the capped term flattens). Dropping the mean-px component from ranking regresses the
 * matrices (blackout geomean 48.57→46.18, mask identity swaps return) — measured 2026-06-11, do not re-walk. */
static bool
association_hypothesis_less(const struct association_pose_hypothesis *a,
                            const struct association_pose_hypothesis *b)
{
	const float cost_a =
	    a->cost.total_nll + a->cost.temporal_nll + a->cost.joint_contention_delta_nll + a->cost.fit_quality_nll;
	const float cost_b =
	    b->cost.total_nll + b->cost.temporal_nll + b->cost.joint_contention_delta_nll + b->cost.fit_quality_nll;
	if (cost_a != cost_b) {
		return cost_a < cost_b;
	}
	if (a->matched_count != b->matched_count) {
		return a->matched_count > b->matched_count;
	}
	return a->cost.reprojection_nll < b->cost.reprojection_nll;
}

static bool
association_single_view_prior_disagrees(const struct association_pose_hypothesis *hyp)
{
	const struct single_view_prior_gate_params params = {
	    .max_prior_nll = ASSOC_SINGLE_VIEW_PRIOR_MAX_NLL,
	    .max_prior_pos_err_m = ASSOC_SINGLE_VIEW_PRIOR_MAX_DISP_M,
	    .min_prior_nll_for_pos_gate = ASSOC_SINGLE_VIEW_PRIOR_DISP_MIN_NLL,
	};
	const struct single_view_prior_gate_evidence evidence = {
	    .distinct_view_count = association_distinct_view_count(hyp),
	    .prior_nll = hyp != NULL ? hyp->cost.position_prior_nll + hyp->cost.orientation_prior_nll : 0.0,
	    .prior_pos_err_m = hyp != NULL ? m_vec3_len(hyp->score.pos_error) : 0.0,
	    .has_pose = hyp != NULL && (hyp->flags & ASSOC_HYP_HAS_POSE) != 0,
	};
	return single_view_prior_gate_disagrees(&evidence, &params);
}

static float
association_visual_nll(const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL) {
		return INFINITY;
	}
	return hyp->cost.total_nll - hyp->cost.position_prior_nll - hyp->cost.orientation_prior_nll -
	       hyp->cost.head_anchor_nll;
}

static float
association_nonvisual_nll(const struct association_pose_hypothesis *hyp)
{
	return hyp != NULL ? hyp->cost.total_nll - association_visual_nll(hyp) : INFINITY;
}

static bool
association_high_evidence_lock_eligible(const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || (hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0) {
		return false;
	}
	if (hyp->matched_count < 8) {
		return false;
	}
	if (!POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_GOOD)) {
		return false;
	}
	/* A single camera can fit a tight-looking but wrong basin when the prior is stale or already poisoned.
	 * If the pose did not also satisfy the prior position/orientation consistency checks AND agree with
	 * the trusted gravity tilt, require enough LEDs to cover a large fraction of the constellation before
	 * treating it as a 6DoF recovery lock. All three prongs are load-bearing: POSITION rules out a
	 * partner/clutter fit far from the prior (xv1 t+75.9 s: a tilt-agreeing pure-yaw clutter fit 1.2 m off);
	 * ORIENT's per-axis bounds — though railed to MAX_ROT_ERROR under covariance flood — still reject a
	 * single-axis 147 deg yaw flip at the prior's position (the same xv1 episode's second entry); and the
	 * TILT channel closes the railed-ORIENT hole for rotations spread across axes (the 138.9 deg flip's
	 * components (44.8, -48.3, -32.7) deg all passed the railed 60 deg bounds — the t=154.2 s selection
	 * defect, results/selection-bug-20260707) because its sigma clamps to GRAVITY_TILT_TOL and never rails
	 * to vacuity. Prior-drifted genuine recoveries take the m>=12, fully-explained branch instead. */
	if (association_distinct_view_count(hyp) == 1 &&
	    !(POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_POSITION | POSE_MATCH_ORIENT) &&
	      hyp->tilt_prior_nll <= ASSOC_HIGH_EVIDENCE_TILT_TRUST_NLL) &&
	    (hyp->matched_count < 12 || hyp->unmatched_count != 0)) {
		return false;
	}
	const double reproj_per_match = hyp->matched_count > 0 ? hyp->score.reprojection_error / (double)hyp->matched_count
	                                                       : INFINITY;
	if (reproj_per_match > 1.5) {
		return false;
	}
	if (association_visual_nll(hyp) >= ASSOC_VISUAL_HIGH_EVIDENCE_NLL &&
	    hyp->cost.total_nll >= ASSOC_VISUAL_LOCK_HIGH_EVIDENCE_COST) {
		return false;
	}
	return hyp->unmatched_count * 4 <= hyp->matched_count + 4;
}

static bool
association_stale_prior_visual_recovery_eligible(const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || (hyp->flags & ASSOC_HYP_STALE_PRIOR) == 0 ||
	    (hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0) {
		return false;
	}
	if (association_distinct_view_count(hyp) != 1) {
		return false;
	}
	if (hyp->matched_count < ASSOC_STALE_PRIOR_RECOVERY_MIN_MATCHED ||
	    hyp->unmatched_count > ASSOC_STALE_PRIOR_RECOVERY_MAX_UNMATCHED) {
		return false;
	}
	if (hyp->cost.total_nll >= ASSOC_STALE_PRIOR_RECOVERY_MAX_TOTAL_NLL ||
	    association_reprojection_per_match(hyp) > ASSOC_STALE_PRIOR_RECOVERY_MAX_REPROJ_PER_LED) {
		return false;
	}
	return POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_GOOD | POSE_MATCH_LED_IDS);
}

static bool
association_baseline_lock_eligible(const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->matched_count < 4 || !POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_GOOD) ||
	    (hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0) {
		return false;
	}

	if (association_high_evidence_lock_eligible(hyp)) {
		return true;
	}
	if (association_single_view_prior_disagrees(hyp)) {
		return false;
	}

	const bool single_view = association_distinct_view_count(hyp) == 1;
	const bool weak_single_view_pose = single_view && hyp->matched_count <= 5;
	if (single_view && hyp->matched_count <= 6 && hyp->unmatched_count > 0 &&
	    association_reprojection_per_match(hyp) > ASSOC_SINGLE_VIEW_MID_LED_MAX_REPROJ_PER_LED) {
		return false;
	}
	if (single_view && hyp->matched_count <= 7 &&
	    hyp->cost.total_nll >= ASSOC_SINGLE_VIEW_LOW_EVIDENCE_MAX_NLL &&
	    hyp->blob_quality_count == hyp->matched_count &&
	    hyp->blob_brightness_mean < assoc_single_view_clutter_max_brightness &&
	    hyp->blob_area_mean < ASSOC_SINGLE_VIEW_CLUTTER_MAX_AREA &&
	    hyp->blob_var_mean_px2 > assoc_single_view_clutter_min_var_px2) {
		return false;
	}
	const float visual_lock_cost =
	    weak_single_view_pose ? ASSOC_VISUAL_SINGLE_VIEW_LOW_LED_LOCK_COST : ASSOC_VISUAL_LOCK_COST;
	return hyp->cost.total_nll < visual_lock_cost;
}

static bool
association_l2_lock_recoverable(const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL) {
		return false;
	}
	if (association_baseline_lock_eligible(hyp)) {
		return false;
	}
	if (association_single_view_prior_disagrees(hyp)) {
		return false;
	}

	const struct l2_accept_params params = {
	    .min_matched = ASSOC_L2_MIN_MATCHED,
	    .min_distinct_views = ASSOC_L2_MIN_DISTINCT_VIEWS,
	    .max_tilt_rad = ASSOC_L2_MAX_TILT_RAD,
	    .max_reproj_per_led_px2 = ASSOC_L2_MAX_REPROJ_PER_LED_PX2,
	    .max_total_nll = ASSOC_L2_MAX_TOTAL_NLL,
	};
	const struct l2_accept_evidence evidence = {
	    .matched_count = hyp->matched_count,
	    .distinct_view_count = association_distinct_view_count(hyp),
	    .reproj_per_led_px2 = hyp->matched_count > 0 ? hyp->score.reprojection_error / (double)hyp->matched_count
	                                                 : INFINITY,
	    .total_nll = hyp->cost.total_nll,
	    .tilt_error_rad = hyp->tilt_error_rad,
	    .tilt_valid = hyp->tilt_valid,
	    .pose_match_good = POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_GOOD),
	    .already_lock_eligible = false,
	};
	return l2_accept_recoverable(&evidence, &params);
}

static bool
association_lock_eligible(const struct association_pose_hypothesis *hyp)
{
	return association_baseline_lock_eligible(hyp) || association_l2_lock_recoverable(hyp);
}

static double
association_reprojection_per_match(const struct association_pose_hypothesis *hyp)
{
	return hyp != NULL && hyp->matched_count > 0 ? hyp->score.reprojection_error / (double)hyp->matched_count
	                                             : INFINITY;
}

static bool
association_position_only_eligible(const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->matched_count < ASSOC_POSITION_ONLY_MIN_MATCHED) {
		return false;
	}
	if ((hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0 && association_distinct_view_count(hyp) < 2) {
		return false;
	}
	const bool stale_visual_recovery = association_stale_prior_visual_recovery_eligible(hyp);
	if (association_single_view_prior_disagrees(hyp) && !stale_visual_recovery) {
		return false;
	}

	if (association_reprojection_per_match(hyp) > ASSOC_POSITION_ONLY_MAX_REPROJ_PER_LED) {
		return false;
	}

	if (association_distinct_view_count(hyp) == 1 && hyp->matched_count <= 5 &&
	    hyp->cost.total_nll >= ASSOC_VISUAL_SINGLE_VIEW_LOW_LED_LOCK_COST) {
		return false;
	}

	if ((hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0 && hyp->matched_count < 5) {
		return false;
	}

	return POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_POSITION) || stale_visual_recovery;
}

static bool
association_led_fold_eligible(const struct association_pose_hypothesis *hyp)
{
	if ((hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0 && association_distinct_view_count(hyp) < 2) {
		return false;
	}
	if (association_single_view_prior_disagrees(hyp)) {
		return false;
	}
	return hyp != NULL && hyp->matched_count > 0 && hyp->cost.total_nll < ASSOC_VISUAL_AMBIG_COST;
}

static float
association_position_observation_std_m(const struct association_pose_hypothesis *hyp)
{
	float std_m = ASSOC_POSITION_ONLY_BASE_STD_M;
	std_m += (float)association_reprojection_per_match(hyp) * ASSOC_POSITION_ONLY_PER_REPROJ_STD_M;
	if (association_distinct_view_count(hyp) < 2) {
		std_m += ASSOC_POSITION_ONLY_SINGLE_VIEW_INFLATE_M;
	}
	if (hyp->matched_count <= 4) {
		std_m += ASSOC_POSITION_ONLY_FEW_LED_INFLATE_M;
	}
	if ((hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0) {
		std_m += ASSOC_POSITION_ONLY_PARTIAL_INFLATE_M;
	}
	if (std_m < ASSOC_POSITION_ONLY_MIN_STD_M) {
		std_m = ASSOC_POSITION_ONLY_MIN_STD_M;
	}
	if (std_m > ASSOC_POSITION_ONLY_MAX_STD_M) {
		std_m = ASSOC_POSITION_ONLY_MAX_STD_M;
	}
	return std_m;
}

static bool
association_position_only_refreshes_optical_anchor(const struct association_pose_hypothesis *hyp,
                                                   bool used_pnp_position)
{
	if (hyp == NULL) {
		return false;
	}
	if (!used_pnp_position) {
		return true;
	}
	if ((hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0) {
		return false;
	}
	if (association_distinct_view_count(hyp) >= 2) {
		return true;
	}
	if (association_stale_prior_visual_recovery_eligible(hyp)) {
		return true;
	}
	return association_lock_eligible(hyp) && POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_GOOD);
}

static float
association_option_cost(const struct association_pose_hypothesis *hyp, enum association_observation_kind kind)
{
	switch (kind) {
	case ASSOC_OBS_POSE_LOCK: {
		const bool visual_recovery =
		    association_high_evidence_lock_eligible(hyp) &&
		    (association_single_view_prior_disagrees(hyp) ||
		     association_nonvisual_nll(hyp) > ASSOC_VISUAL_RECOVERY_PRIOR_NLL);
		/* Visual recovery re-prices a high-evidence candidate in a "the coasted prior has drifted"
		 * world: the position, yaw and head-anchor prior objections are waived for a flat action fee,
		 * because those DoFs genuinely drift through a dropout (magnetometer-less yaw above all). The
		 * GRAVITY-TILT channel is never waived: tilt is gravity-anchored and driftless (the accel pins
		 * the prior's tilt through any dropout — the same physics that keeps prior_tilt_trusted set),
		 * so a tilt disagreement is evidence against the CANDIDATE, not the prior. It is charged
		 * softly capped (recoverable, not a veto) and the re-price is an option — never above the
		 * honest full-prior evaluation. Waiving tilt too turned the waiver into a gift of
		 * prior_cost - 2 nats that GREW with how wrong the candidate was, selecting 139-145 deg flips
		 * over concurrent cheaper uprights (results/selection-bug-20260707). */
		float base_nll = hyp->cost.total_nll;
		if (visual_recovery) {
			float tilt_nll = hyp->tilt_prior_nll;
			if (tilt_nll > ASSOC_VISUAL_RECOVERY_TILT_MAX_NLL) {
				tilt_nll = ASSOC_VISUAL_RECOVERY_TILT_MAX_NLL;
			}
			const float reprice_nll =
			    association_visual_nll(hyp) + tilt_nll + ASSOC_VISUAL_RECOVERY_ACTION_NLL;
			if (reprice_nll < base_nll) {
				base_nll = reprice_nll;
			}
		}
		float evidence_nll = ASSOC_LOCK_MATCH_EVIDENCE_NLL * (float)hyp->matched_count;
		if (association_distinct_view_count(hyp) >= 2) {
			evidence_nll += ASSOC_LOCK_MULTIVIEW_EVIDENCE_NLL;
		}
		if (POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_STRONG)) {
			evidence_nll += ASSOC_LOCK_STRONG_EVIDENCE_NLL;
		}
		if (evidence_nll > ASSOC_LOCK_MAX_EVIDENCE_NLL) {
			evidence_nll = ASSOC_LOCK_MAX_EVIDENCE_NLL;
		}
		const float recovery_nll = association_l2_lock_recoverable(hyp) ? ASSOC_L2_RECOVERY_ACTION_NLL : 0.0f;
		return base_nll + hyp->cost.temporal_nll + hyp->cost.orientation_consensus_nll + recovery_nll +
		       hyp->cost.joint_contention_delta_nll + hyp->cost.fit_quality_nll - evidence_nll;
	}
	case ASSOC_OBS_POSITION_ONLY:
		return hyp->cost.total_nll + hyp->cost.temporal_nll + hyp->cost.joint_contention_delta_nll +
		       hyp->cost.fit_quality_nll + ASSOC_POSITION_ONLY_ACTION_NLL;
	case ASSOC_OBS_LED_FOLD:
		return hyp->cost.total_nll + hyp->cost.temporal_nll + hyp->cost.joint_contention_delta_nll +
		       hyp->cost.fit_quality_nll + ASSOC_LED_FOLD_ACTION_NLL;
	case ASSOC_OBS_ABSENT:
	default: return 0.0f;
	}
}

static void
association_insert_hypothesis(struct association_device_work *work,
                              const struct association_pose_hypothesis *candidate)
{
	for (int i = 0; i < work->count; i++) {
		if (association_pose_duplicate(&work->hyps[i], candidate)) {
			if (association_hypothesis_less(candidate, &work->hyps[i])) {
				work->hyps[i] = *candidate;
			}
			/* KNOWN, CALIBRATED-IN (audit 2026-07-10 R1-M1): the improved entry is NOT
			 * re-bubbled, so the array can transiently violate the sort invariant and
			 * first-eligible consumers see the stale order. The one-line bubble fix was
			 * implemented and REFUTED on the standing regression gate (left yield -6.89,
			 * a post-coast re-acquisition never happens at t=112.218; eyes-on record in
			 * results/w2-fold-20260711/m1-insert-order-adjudication/). The selection
			 * stack is calibrated WITH this order; re-open only as its own matrix-gated
			 * experiment inside the W5 rework of the first-eligible consumers. */
			return;
		}
	}

	if (work->count < ASSOCIATION_MAX_HYPOTHESES_PER_DEVICE) {
		work->hyps[work->count++] = *candidate;
	} else if (association_hypothesis_less(candidate, &work->hyps[work->count - 1])) {
		work->hyps[work->count - 1] = *candidate;
	}

	for (int i = work->count - 1; i > 0 && association_hypothesis_less(&work->hyps[i], &work->hyps[i - 1]); i--) {
		struct association_pose_hypothesis tmp = work->hyps[i - 1];
		work->hyps[i - 1] = work->hyps[i];
		work->hyps[i] = tmp;
	}
}

static double
association_gravity_tilt_delta_rad(const struct xrt_quat *candidate_cam,
                                   const struct xrt_pose *P_world_obj_ref,
                                   const struct tracking_sample_frame *view)
{
	struct xrt_pose P_cam_obj_ref;
	math_pose_transform(&view->P_cam_world, P_world_obj_ref, &P_cam_obj_ref);

	struct xrt_quat candidate_inv;
	struct xrt_quat ref_inv;
	math_quat_invert(candidate_cam, &candidate_inv);
	math_quat_invert(&P_cam_obj_ref.orientation, &ref_inv);

	struct xrt_vec3 candidate_up;
	struct xrt_vec3 ref_up;
	math_quat_rotate_vec3(&candidate_inv, &view->cam_gravity_vector, &candidate_up);
	math_quat_rotate_vec3(&ref_inv, &view->cam_gravity_vector, &ref_up);

	double dot = (double)candidate_up.x * ref_up.x + (double)candidate_up.y * ref_up.y +
	             (double)candidate_up.z * ref_up.z;
	dot = dot > 1.0 ? 1.0 : (dot < -1.0 ? -1.0 : dot);
	return acos(dot);
}

static int
association_collect_matched_points(const struct pose_metrics_blob_match_info *match_info,
                                   struct xrt_vec3 obj_pts[MAX_OBJECT_LEDS],
                                   struct xrt_vec2 img_pts[MAX_OBJECT_LEDS])
{
	int n = 0;
	for (int i = 0; i < match_info->num_visible_leds && n < MAX_OBJECT_LEDS; i++) {
		const struct pose_metrics_visible_led_info *visible = &match_info->visible_leds[i];
		if (visible->matched_blob == NULL || visible->led == NULL) {
			continue;
		}
		obj_pts[n] = visible->led->pos;
		img_pts[n].x = visible->matched_blob->x;
		img_pts[n].y = visible->matched_blob->y;
		n++;
	}
	return n;
}

static bool
association_add_pose_hypothesis(struct t_constellation_tracker *ct,
                                struct association_device_work *work,
                                struct tracking_sample_device_state *dev_state,
                                struct constellation_tracking_sample *sample,
                                int view_id,
                                enum association_hypothesis_source source,
                                uint16_t flags,
                                const struct xrt_pose *P_cam_obj,
                                bool allow_refine)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;

	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return false;
	}

	const bool ignore_prior = (flags & ASSOC_HYP_IGNORE_PRIOR) != 0;
	const bool use_prior = dev_state->prior_tilt_trusted && !ignore_prior;
	const bool use_gravity_ref = use_prior && dev_state->gravity_ref_valid && dev_state->gravity_ref_clean;
	struct xrt_pose P_cam_obj_local = *P_cam_obj;
	const double tilt_delta_rad =
	    use_gravity_ref
	        ? association_gravity_tilt_delta_rad(&P_cam_obj_local.orientation, &dev_state->P_world_obj_gravity,
	                                             view)
	    : use_prior ? association_gravity_tilt_delta_rad(&P_cam_obj_local.orientation,
	                                                     &dev_state->P_world_obj_prior, view)
	                : 0.0;

	if ((use_gravity_ref || use_prior) && tilt_delta_rad > ASSOC_TILT_CLAMP_MIN_DELTA_RAD) {
		struct xrt_pose P_cam_obj_ref;
		math_pose_transform(&view->P_cam_world,
		                    use_gravity_ref ? &dev_state->P_world_obj_gravity : &dev_state->P_world_obj_prior,
		                    &P_cam_obj_ref);

		struct pose_metrics_blob_match_info before;
		pose_metrics_match_pose_to_blobs(&P_cam_obj_local, view->bwobs->blobs, view->bwobs->num_blobs,
		                                 &device->led_model, &cam->camera_model, &before);
		if (before.matched_blobs >= 4) {
			struct xrt_vec3 obj_pts[MAX_OBJECT_LEDS];
			struct xrt_vec2 img_pts[MAX_OBJECT_LEDS];
			const int n = association_collect_matched_points(&before, obj_pts, img_pts);
			struct xrt_pose clamped;
			if (n >= 4 && ransac_pnp_tilt_clamp(&P_cam_obj_local, obj_pts, img_pts, n, &cam->camera_model,
			                                    &P_cam_obj_ref, &view->cam_gravity_vector,
			                                    !use_gravity_ref, &clamped, NULL, NULL)) {
				struct pose_metrics_blob_match_info after;
				pose_metrics_match_pose_to_blobs(&clamped, view->bwobs->blobs, view->bwobs->num_blobs,
				                                 &device->led_model, &cam->camera_model, &after);
				const double reproj_per_led =
				    after.matched_blobs > 0 ? after.reprojection_error / after.matched_blobs : INFINITY;
				if (after.matched_blobs >= 4 && after.matched_blobs + 1 >= before.matched_blobs &&
				    reproj_per_led <= ASSOC_TILT_CLAMP_MAX_REPROJ_PER_LED) {
					P_cam_obj_local = clamped;
				}
			}
		}
	}
	P_cam_obj = &P_cam_obj_local;

	if (use_gravity_ref && (flags & ASSOC_HYP_IS_TWIN) == 0) {
		struct xrt_pose P_cam_obj_ref;
		math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_gravity, &P_cam_obj_ref);
		struct pose_metrics_blob_match_info match_info;
		pose_metrics_match_pose_to_blobs(&P_cam_obj_local, view->bwobs->blobs, view->bwobs->num_blobs,
		                                 &device->led_model, &cam->camera_model, &match_info);
		if (match_info.matched_blobs >= 4) {
			struct xrt_vec3 obj_pts[MAX_OBJECT_LEDS];
			struct xrt_vec2 img_pts[MAX_OBJECT_LEDS];
			const int n = association_collect_matched_points(&match_info, obj_pts, img_pts);
			struct xrt_pose clamped;
			struct xrt_pose yaw_twin;
			bool has_yaw_twin = false;
			if (n >= 4 &&
			    ransac_pnp_tilt_clamp(P_cam_obj, obj_pts, img_pts, n, &cam->camera_model, &P_cam_obj_ref,
			                          &view->cam_gravity_vector, true, &clamped, &yaw_twin, &has_yaw_twin) &&
			    has_yaw_twin) {
				association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source,
				                                (uint16_t)(flags | ASSOC_HYP_HAS_TWIN | ASSOC_HYP_IS_TWIN),
				                                &yaw_twin, false);
			}
		}
	}

	struct association_pose_hypothesis hyp;
	association_hypothesis_init(&hyp, (uint8_t)dev_state->dev_index, (uint8_t)source);
	hyp.flags = flags | ASSOC_HYP_HAS_POSE;
	if (!dev_state->prior_position_tracked) {
		hyp.flags |= ASSOC_HYP_PRIOR_POSITION_UNTRACKED;
	}
	if (dev_state->prior_optical_stale) {
		hyp.flags |= ASSOC_HYP_STALE_PRIOR;
	}
	hyp.primary_view_id = (int16_t)view_id;
	hyp.pose_cam = *P_cam_obj;
	math_pose_transform(&view->P_world_cam, P_cam_obj, &hyp.pose_world);
	math_pose_transform(&cam->P_imu_cam, P_cam_obj, &hyp.pose_imu);

	struct pose_metrics primary_score = {0};
	bool have_primary_score = false;
	bool saw_partial_only = false;
	int total_visible = 0;
	int total_unmatched = 0;
	double total_reprojection = 0.0;
	/* Detection-likelihood terms summed over views with at least two plausible matches. Empty sibling views
	 * are not hard negative evidence: edge-FOV and occlusion frames can be legitimate single-view observations. */
	double total_data_nll_detection = 0.0;
	double total_fit_nll = 0.0;
	double total_retention_nll = 0.0;
	double total_data_nll_missed_if_matched = 0.0;

	for (int match_view_id = 0; match_view_id < sample->n_views; match_view_id++) {
		struct tracking_sample_frame *match_view = sample->views + match_view_id;
		struct constellation_tracker_camera_state *match_cam = ct->cam + match_view_id;
		blobservation *bwobs = match_view->bwobs;
		if (bwobs == NULL || bwobs->num_blobs == 0) {
			continue;
		}

		struct xrt_pose P_match_cam_obj;
		math_pose_transform(&match_view->P_cam_world, &hyp.pose_world, &P_match_cam_obj);

		struct xrt_pose P_match_cam_obj_prior;
		math_pose_transform(&match_view->P_cam_world, &dev_state->P_world_obj_prior,
		                    &P_match_cam_obj_prior);

		struct pose_metrics score;
		struct pose_metrics_blob_match_info match_info;
		if (use_prior) {
			// The evaluation computes the full match internally; take it instead of
			// re-running the project+gate+GNN pass (one of the measured hot duplicates).
			pose_metrics_evaluate_pose_with_prior(&score, &P_match_cam_obj, false,
			                                      &P_match_cam_obj_prior, &dev_state->prior_pos_error,
			                                      &dev_state->prior_rot_error, bwobs->blobs,
			                                      bwobs->num_blobs, &device->led_model,
			                                      &match_cam->camera_model, NULL, &match_info);
		} else {
			pose_metrics_evaluate_pose(&score, &P_match_cam_obj, bwobs->blobs, bwobs->num_blobs,
			                           &device->led_model, &match_cam->camera_model, NULL);
			pose_metrics_match_pose_to_blobs(&P_match_cam_obj, bwobs->blobs, bwobs->num_blobs,
			                                 &device->led_model, &match_cam->camera_model, &match_info);
		}
		if (score.matched_blobs < 2) {
			continue;
		}
		total_data_nll_detection +=
		    match_info.data_nll_detection -
		    association_partner_occlusion_credit(sample, dev_state, match_view, match_cam, &match_info);
		total_data_nll_missed_if_matched += match_info.data_nll_missed_if_matched;
		for (int li = 0; li < match_info.num_visible_leds; li++) {
			const struct pose_metrics_visible_led_info *vled = &match_info.visible_leds[li];
			if (vled->matched_blob == NULL) {
				continue;
			}
			const double fdx = vled->pos_px.x - vled->matched_blob->x;
			const double fdy = vled->pos_px.y - vled->matched_blob->y;
			total_fit_nll += 0.5 * ASSOC_FIT_CAUCHY_K2 *
			                 log1p((fdx * fdx + fdy * fdy) / (ASSOC_FIT_SIGMA2_PX2 * ASSOC_FIT_CAUCHY_K2));
			if (vled->matched_blob->retention_class == BLOB_RETENTION_STATIC_CLUTTER) {
				double ramp = ((double)vled->matched_blob->static_dwell_s - STATIC_MAP_STATIC_DWELL_S) /
				              (ASSOC_RETENTION_DWELL_SAT_S - STATIC_MAP_STATIC_DWELL_S);
				ramp = ramp < 0.0 ? 0.0 : (ramp > 1.0 ? 1.0 : ramp);
				total_retention_nll += ASSOC_RETENTION_STATIC_NLL * ramp;
			}
		}

		const double view_error_per_blob = score.matched_blobs > 0
		                                       ? score.reprojection_error / (double)score.matched_blobs
		                                       : INFINITY;
		if (match_view_id != view_id && view_error_per_blob > 4.0) {
			continue;
		}

		const int added = association_fill_blob_refs(&hyp, &match_info, bwobs, match_view_id);
		if (added == 0) {
			continue;
		}

		total_visible += score.visible_leds;
		total_unmatched += score.unmatched_blobs;
		total_reprojection += score.reprojection_error;
		saw_partial_only |= POSE_HAS_FLAGS(&score, POSE_MATCH_PRIOR_SUPPORTED_PARTIAL);
		if (!have_primary_score || match_view_id == view_id) {
			primary_score = score;
			have_primary_score = true;
		}
	}

	if (!have_primary_score || hyp.matched_count < 2) {
		return false;
	}

	if (allow_refine) {
		struct xrt_pose refined_pose;
		if (association_refine_multiview_pose(ct, dev_state, sample, &hyp, &refined_pose)) {
			association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source, flags,
			                                &refined_pose, false);
		}
	}

	hyp.visible_count = (uint8_t)(total_visible > UINT8_MAX ? UINT8_MAX : total_visible);
	hyp.unmatched_count = (uint8_t)(total_unmatched > UINT8_MAX ? UINT8_MAX : total_unmatched);
	hyp.score = primary_score;
	hyp.score.visible_leds = hyp.visible_count;
	hyp.score.matched_blobs = hyp.matched_count;
	hyp.score.unmatched_blobs = hyp.unmatched_count;
	hyp.score.reprojection_error = total_reprojection;
	association_update_blob_quality(&hyp, sample);

	const float matched = hyp.matched_count > 0 ? (float)hyp.matched_count : 1.0f;
	const float error_per_observation = (float)(total_reprojection / (double)matched);
	const bool multi_view = association_distinct_view_count(&hyp) >= 2;
	if (hyp.matched_count >= 5 && error_per_observation < 2.0f &&
	    (multi_view || POSE_HAS_FLAGS(&primary_score, POSE_MATCH_POSITION | POSE_MATCH_ORIENT))) {
		hyp.score.match_flags |= POSE_MATCH_GOOD;
		hyp.score.match_flags &= ~POSE_MATCH_PRIOR_SUPPORTED_PARTIAL;
		if (error_per_observation < 0.75f && hyp.matched_count >= 6) {
			hyp.score.match_flags |= POSE_MATCH_STRONG;
		}
	} else if (saw_partial_only) {
		hyp.flags |= ASSOC_HYP_PARTIAL_ONLY;
		hyp.score.match_flags |= POSE_MATCH_PRIOR_SUPPORTED_PARTIAL;
	}
	/* Detection likelihood replaces fixed missed-LED penalties and matched-blob rewards. Reprojection remains
	 * a per-observation fit-quality term; detection_ref carries the cardinality preference on the same NLL scale. */
	const double detection_ref = total_data_nll_detection - total_data_nll_missed_if_matched;
	hyp.cost.reprojection_nll = error_per_observation;
	hyp.cost.missed_led_nll = (float)detection_ref;
	hyp.cost.clutter_nll = (float)total_unmatched * ASSOC_CLUTTER_NLL;
	hyp.cost.position_prior_nll = association_position_prior_nll(&primary_score, dev_state);
	if (dev_state->prior_tilt_trusted) {
		/* One swing-twist decompose prices BOTH channels: the full anisotropic prior and the
		 * tilt-only charge the recovery re-price keeps unwaived. The stored split is also the
		 * single source for candidate telemetry (association_emit_candidate). */
		struct xrt_pose P_cam_obj_prior;
		math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);
		double tilt_rad = 0.0;
		double yaw_rad = 0.0;
		pose_metrics_prior_orient_split(&P_cam_obj->orientation, &P_cam_obj_prior.orientation,
		                                &view->cam_gravity_vector, &tilt_rad, &yaw_rad);
		hyp.cost.orientation_prior_nll = (float)pose_metrics_prior_orient_cost_from_split(
		    tilt_rad, yaw_rad, dev_state->prior_tilt_sigma_rad, dev_state->prior_yaw_sigma_rad,
		    FLIP_COST_HUBER_KNEE_SIGMA, FLIP_COST_WEIGHT);
		hyp.tilt_error_rad = (float)tilt_rad;
		hyp.yaw_error_rad = (float)yaw_rad;
		hyp.tilt_prior_nll = (float)pose_metrics_prior_orient_cost_from_split(
		    tilt_rad, 0.0, dev_state->prior_tilt_sigma_rad, 0.0, FLIP_COST_HUBER_KNEE_SIGMA,
		    FLIP_COST_WEIGHT);
		hyp.tilt_valid = true;
	}
	hyp.cost.head_anchor_nll = association_head_anchor_nll(device, cam, view, sample, P_cam_obj);
	hyp.cost.fit_quality_nll = (float)total_fit_nll;
	hyp.cost.retention_nll = (float)(total_retention_nll < ASSOC_RETENTION_TOTAL_MAX_NLL
	                                     ? total_retention_nll
	                                     : (double)ASSOC_RETENTION_TOTAL_MAX_NLL);
	hyp.cost.temporal_nll = association_temporal_yaw_nll(device, dev_state, sample->timestamp, &hyp.pose_world);
	hyp.cost.total_nll = hyp.cost.reprojection_nll + hyp.cost.missed_led_nll + hyp.cost.clutter_nll +
	                     hyp.cost.retention_nll + hyp.cost.position_prior_nll + hyp.cost.orientation_prior_nll +
	                     hyp.cost.head_anchor_nll;

	if ((hyp.flags & ASSOC_HYP_PARTIAL_ONLY) != 0 && association_distinct_view_count(&hyp) < 2) {
		return false;
	}

	association_insert_hypothesis(work, &hyp);
	association_emit_candidate(device, dev_state, view, view_id, sample->timestamp, &hyp, false, 0, NULL);
	return true;
}

static void
association_add_prior_pose_sources(struct t_constellation_tracker *ct,
                                   struct association_device_work *work,
                                   struct tracking_sample_device_state *dev_state,
                                   struct constellation_tracking_sample *sample)
{
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}
		work->has_blobs = true;
		struct xrt_pose P_cam_obj;
		if (dev_state->prior_tilt_trusted) {
			math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj);
			association_add_pose_hypothesis(ct, work, dev_state, sample, view_id,
			                                ASSOC_SOURCE_PRIOR_POSE, ASSOC_HYP_NONE, &P_cam_obj, true);
		}
		if (dev_state->have_last_seen_pose) {
			math_pose_transform(&view->P_cam_world, &dev_state->last_seen_pose, &P_cam_obj);
			association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, ASSOC_SOURCE_LAST_SEEN,
			                                ASSOC_HYP_NONE, &P_cam_obj, true);
		}
	}
}

static void
association_add_labelled_pnp_source(struct t_constellation_tracker *ct,
                                    struct association_device_work *work,
                                    struct tracking_sample_device_state *dev_state,
                                    struct constellation_tracking_sample *sample,
                                    int view_id,
                                    enum association_hypothesis_source source)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	blobservation *bwobs = view->bwobs;

	const int labelled_count = association_count_labelled_blobs(bwobs, device->led_model.id);
	if (labelled_count < 3) {
		return;
	}

	if (labelled_count == 3) {
		struct xrt_pose p3p_solutions[4];
		const int n_solutions = pnp_solve_p3p(bwobs->blobs, bwobs->num_blobs, &device->led_model,
		                                      &cam->camera_model, p3p_solutions,
		                                      (int)ARRAY_SIZE(p3p_solutions));
		if (source == ASSOC_SOURCE_PRIOR_LABELLED_PNP && dev_state->prior_tilt_trusted) {
			int best = -1;
			float best_nll = INFINITY;
			for (int i = 0; i < n_solutions; i++) {
				const float nll =
				    association_prior_pose_nll(device, cam, dev_state, view, sample, &p3p_solutions[i]);
				if (nll < best_nll) {
					best_nll = nll;
					best = i;
				}
			}
			if (best >= 0) {
				association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source,
				                                ASSOC_HYP_HAS_TWIN, &p3p_solutions[best], false);
			}
			return;
		}
		for (int i = 0; i < n_solutions; i++) {
			association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source,
			                                ASSOC_HYP_HAS_TWIN, &p3p_solutions[i], true);
		}
		return;
	}

	struct xrt_pose P_cam_obj;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj);
	struct xrt_pose twin_pose;
	bool has_twin = false;
	if (!ransac_pnp_pose_with_twin(&P_cam_obj, bwobs->blobs, bwobs->num_blobs, &device->led_model,
	                               &cam->camera_model, NULL, NULL, &twin_pose, &has_twin)) {
		return;
	}

	const uint16_t twin_flag = has_twin ? ASSOC_HYP_HAS_TWIN : ASSOC_HYP_NONE;
	association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source, twin_flag, &P_cam_obj, true);
	if (has_twin) {
		association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source,
		                                ASSOC_HYP_HAS_TWIN | ASSOC_HYP_IS_TWIN, &twin_pose, true);
	}
}

static void
association_add_label_sources(struct t_constellation_tracker *ct,
                              struct association_device_work *work,
                              struct tracking_sample_device_state *dev_state,
                              struct constellation_tracking_sample *sample)
{
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}

		association_add_labelled_pnp_source(ct, work, dev_state, sample, view_id, ASSOC_SOURCE_LABELLED_PNP);

		struct association_blob_label_snapshot snapshot;
		association_save_labels(view->bwobs, &snapshot);
		const int n_labelled = device_propagate_labels_in_view(ct, dev_state, sample, view_id);
		if (n_labelled >= 3) {
			association_add_labelled_pnp_source(ct, work, dev_state, sample, view_id,
			                                    ASSOC_SOURCE_PRIOR_LABELLED_PNP);
		}
		association_restore_labels(view->bwobs, &snapshot);
	}
}

static void
association_add_joint_pnp_source(struct t_constellation_tracker *ct,
                                 struct association_device_work *work,
                                 struct tracking_sample_device_state *dev_state,
                                 struct constellation_tracking_sample *sample)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct association_blob_label_snapshot snapshots[CONSTELLATION_MAX_CAMERAS];
	bool have_snapshot[CONSTELLATION_MAX_CAMERAS] = {false};
	struct joint_pnp_view jviews[JOINT_PNP_MAX_VIEWS];
	int view_ids[JOINT_PNP_MAX_VIEWS];
	int n_jviews = 0;

	for (int view_id = 0; view_id < sample->n_views && n_jviews < JOINT_PNP_MAX_VIEWS; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}
		association_save_labels(view->bwobs, &snapshots[view_id]);
		have_snapshot[view_id] = true;
		const int n_labelled = device_propagate_labels_in_view(ct, dev_state, sample, view_id);
		if (n_labelled < 3) {
			continue;
		}
		jviews[n_jviews].blobs = view->bwobs->blobs;
		jviews[n_jviews].num_blobs = view->bwobs->num_blobs;
		jviews[n_jviews].calib = &cam->camera_model;
		jviews[n_jviews].P_imu_cam = cam->P_imu_cam;
		view_ids[n_jviews] = view_id;
		n_jviews++;
	}

	if (n_jviews >= 2) {
		const int seed_view = view_ids[0];
		struct constellation_tracker_camera_state *seed_cam = ct->cam + seed_view;
		struct tracking_sample_frame *seed = sample->views + seed_view;
		struct xrt_pose P_cam_obj_prior, P_imu_obj;
		math_pose_transform(&seed->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);
		math_pose_transform(&seed_cam->P_imu_cam, &P_cam_obj_prior, &P_imu_obj);

		int num_rays = 0, num_inliers = 0;
		if (joint_pnp_solve(&P_imu_obj, jviews, n_jviews, &device->led_model, &num_rays, &num_inliers)) {
			for (int i = 0; i < n_jviews; i++) {
				const int view_id = view_ids[i];
				struct xrt_pose P_cam_imu, P_cam_obj;
				math_pose_invert(&ct->cam[view_id].P_imu_cam, &P_cam_imu);
				math_pose_transform(&P_cam_imu, &P_imu_obj, &P_cam_obj);
				association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, ASSOC_SOURCE_JOINT_PNP,
				                                ASSOC_HYP_JOINT, &P_cam_obj, true);
			}
		}
	}

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		if (have_snapshot[view_id]) {
			association_restore_labels(sample->views[view_id].bwobs, &snapshots[view_id]);
		}
	}
}

static int
association_prior_visible_led_bounds(const struct tracking_sample_device_state *dev_state,
                                     const struct tracking_sample_frame *view,
                                     const struct constellation_tracker_camera_state *cam,
                                     float margin_px,
                                     struct pose_rect *out_bounds)
{
	if (dev_state == NULL || view == NULL || cam == NULL || dev_state->led_model == NULL) {
		return 0;
	}

	struct xrt_pose P_cam_obj;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj);

	int visible = 0;
	const struct t_constellation_led_model *led_model = dev_state->led_model;
	for (int i = 0; i < led_model->num_leds; i++) {
		const struct t_constellation_led *led = led_model->leds + i;
		struct xrt_vec3 led_cam;
		math_pose_transform_point(&P_cam_obj, &led->pos, &led_cam);
		if (led_cam.z <= 0.0f) {
			continue;
		}

		struct xrt_vec2 px;
		if (!t_camera_models_project(&cam->camera_model.calib, led_cam.x, led_cam.y, led_cam.z, &px.x, &px.y)) {
			continue;
		}
		if (px.x < -margin_px || px.y < -margin_px || px.x >= (float)cam->camera_model.width + margin_px ||
		    px.y >= (float)cam->camera_model.height + margin_px) {
			continue;
		}

		struct xrt_vec3 view_vec = led_cam;
		struct xrt_vec3 normal;
		math_vec3_normalize(&view_vec);
		math_quat_rotate_vec3(&P_cam_obj.orientation, &led->dir, &normal);
		if (m_vec3_dot(view_vec, normal) > cos(DEG_TO_RAD(180.0 - LED_ANGLE))) {
			continue;
		}
		if (out_bounds != NULL) {
			if (visible == 0) {
				*out_bounds = (struct pose_rect){.left = px.x, .right = px.x, .top = px.y, .bottom = px.y};
			} else {
				if (px.x < out_bounds->left) {
					out_bounds->left = px.x;
				}
				if (px.x > out_bounds->right) {
					out_bounds->right = px.x;
				}
				if (px.y < out_bounds->top) {
					out_bounds->top = px.y;
				}
				if (px.y > out_bounds->bottom) {
					out_bounds->bottom = px.y;
				}
			}
		}
		visible++;
	}
	return visible;
}

static int
association_count_prior_visible_leds_in_view(const struct tracking_sample_device_state *dev_state,
                                             const struct tracking_sample_frame *view,
                                             const struct constellation_tracker_camera_state *cam)
{
	return association_prior_visible_led_bounds(dev_state, view, cam, 64.0f, NULL);
}

/* Cold-search blob staging order IS the anchor priority: correspondence search tries anchor
 * blobs in array order inside every LED-combination pass, so under the per-frame work budget
 * a stable partition with STATIC_CLUTTER blobs last spends trials on live candidates first
 * and falls through to positively-static blobs only when the live ones fail. Suppression
 * never removes a blob from the set. */
static void
association_partition_static_last(struct blob *blobs, int n)
{
	struct blob staged[MAX_BLOBS_PER_FRAME];
	int n_live = 0, n_static = 0;
	for (int i = 0; i < n; i++) {
		if (blobs[i].retention_class == BLOB_RETENTION_STATIC_CLUTTER) {
			staged[n_static++] = blobs[i];
		} else {
			blobs[n_live++] = blobs[i];
		}
	}
	memcpy(blobs + n_live, staged, (size_t)n_static * sizeof(struct blob));
}

/* Snapshot a view's blobs for a full-frame cold-search task, static-clutter last. The copy also
 * decouples the task from concurrent label updates on the live observation. */
static int
association_stage_search_blobs(const struct tracking_sample_frame *view, struct blob *out_blobs)
{
	const int n = view->bwobs->num_blobs < MAX_BLOBS_PER_FRAME ? view->bwobs->num_blobs : MAX_BLOBS_PER_FRAME;
	memcpy(out_blobs, view->bwobs->blobs, (size_t)n * sizeof(struct blob));
	association_partition_static_last(out_blobs, n);
	return n;
}

struct association_prior_roi_result
{
	enum g2_search_result status;
	int blob_count;
	int visible_leds;
};

static struct association_prior_roi_result
association_build_prior_roi_blobs(const struct tracking_sample_device_state *dev_state,
                                  const struct tracking_sample_frame *view,
                                  struct constellation_tracker_camera_state *cam,
                                  struct blob out_blobs[MAX_BLOBS_PER_FRAME])
{
	struct association_prior_roi_result result = {
	    .status = G2_SEARCH_ROI_SKIP_INVALID_PRIOR,
	    .blob_count = 0,
	    .visible_leds = 0,
	};
	if (dev_state == NULL || view == NULL || cam == NULL || view->bwobs == NULL || view->bwobs->num_blobs < 4) {
		result.status = G2_SEARCH_ROI_SKIP_NO_BLOBS;
		return result;
	}
	if (dev_state->led_model == NULL) {
		result.status = G2_SEARCH_ROI_SKIP_NO_MODEL;
		return result;
	}
	if (!dev_state->prior_tilt_trusted) {
		result.status = G2_SEARCH_ROI_SKIP_UNTRUSTED_PRIOR;
		return result;
	}
	if (dev_state->prior_pos_error.x <= 0.0f) {
		result.status = G2_SEARCH_ROI_SKIP_INVALID_PRIOR;
		return result;
	}

	struct pose_rect bounds = {0};
	const int visible = association_prior_visible_led_bounds(dev_state, view, cam, 64.0f, &bounds);
	result.visible_leds = visible;
	if (visible < 3) {
		result.status = G2_SEARCH_ROI_SKIP_VISIBLE_LT3;
		return result;
	}

	const float focal_px = (float)MAX(cam->camera_model.calib.fx, cam->camera_model.calib.fy);
	struct xrt_pose P_cam_obj;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj);
	const float depth_m = fmaxf(P_cam_obj.position.z, 0.10f);
	const float pos_sigma_m = fmaxf(dev_state->prior_pos_error.x,
	                                fmaxf(dev_state->prior_pos_error.y, dev_state->prior_pos_error.z));
	const float rot_sigma_rad = fmaxf(dev_state->prior_rot_error.x,
	                                  fmaxf(dev_state->prior_rot_error.y, dev_state->prior_rot_error.z));
	const float uncertainty_px =
	    focal_px * (pos_sigma_m + ASSOC_COLD_PRIOR_ROI_ROT_LEVER_M * rot_sigma_rad) / depth_m;
	const float margin_px =
	    fminf(fmaxf(ASSOC_COLD_PRIOR_ROI_MIN_MARGIN_PX, uncertainty_px + 48.0f),
	          ASSOC_COLD_PRIOR_ROI_MAX_MARGIN_PX);

	bounds.left -= margin_px;
	bounds.top -= margin_px;
	bounds.right += margin_px;
	bounds.bottom += margin_px;

	int n = 0;
	for (int i = 0; i < view->bwobs->num_blobs && n < MAX_BLOBS_PER_FRAME; i++) {
		const struct blob *b = view->bwobs->blobs + i;
		if (b->x < bounds.left || b->x > bounds.right || b->y < bounds.top || b->y > bounds.bottom) {
			continue;
		}
		out_blobs[n++] = *b;
	}

	result.blob_count = n;
	if (n < 4) {
		result.status = G2_SEARCH_ROI_SKIP_BLOBS_LT4;
		return result;
	}
	if (n >= view->bwobs->num_blobs) {
		result.status = G2_SEARCH_ROI_SKIP_FULL_EQUIV;
		return result;
	}
	association_partition_static_last(out_blobs, n);
	result.status = G2_SEARCH_SUCCESS;
	return result;
}

/* Wave-C suppression needs STRONG evidence, not merely usable: a mirror flip can be lock/fold
 * eligible (GOOD, plausible reprojection) and would then suppress the full-frame search that
 * disambiguates it — the baseline found the true pose at hand-GT frame cam0_150718252458505
 * via exactly that search while a GOOD-flip-gated wave C reproduced a 135 deg flip there. */
static bool
association_work_has_strong_hypothesis(const struct association_device_work *work)
{
	for (int i = 0; work != NULL && i < work->count; i++) {
		if (POSE_HAS_FLAGS(&work->hyps[i].score, POSE_MATCH_STRONG)) {
			return true;
		}
	}
	return false;
}

static bool
association_work_has_view_hypothesis(const struct association_device_work *work, int view_id)
{
	for (int i = 0; work != NULL && i < work->count; i++) {
		if (association_hypothesis_has_view(&work->hyps[i], view_id)) {
			return true;
		}
	}
	return false;
}

/* Blobs already labelled as ANOTHER device's LEDs must not make a view look "uncovered" for
 * this device — at hands-close/table scenarios that fired the deep cold search ~0.9x per
 * device-frame on the partner's constellation (measured 64.6M of 198M P3P trials). */
static int
association_view_unclaimed_blob_count(const struct tracking_sample_frame *view, int self_model_id)
{
	int count = 0;
	for (int i = 0; i < view->bwobs->num_blobs; i++) {
		const int led_id = view->bwobs->blobs[i].led_id;
		if (led_id != LED_INVALID_ID && LED_OBJECT_ID(led_id) != self_model_id) {
			continue;
		}
		count++;
	}
	return count;
}

static bool
association_work_has_uncovered_blob_view(const struct association_device_work *work,
                                         const struct constellation_tracking_sample *sample,
                                         int self_model_id)
{
	for (int view_id = 0; sample != NULL && view_id < sample->n_views; view_id++) {
		const struct tracking_sample_frame *view = &sample->views[view_id];
		if (view->bwobs == NULL ||
		    association_view_unclaimed_blob_count(view, self_model_id) < 4) {
			continue;
		}
		if (!association_work_has_view_hypothesis(work, view_id)) {
			return true;
		}
	}
	return false;
}

static void
association_resort_work(struct association_device_work *work);

static enum correspondence_search_flags
association_cold_search_flags(const struct tracking_sample_device_state *dev_state,
                              int search_blob_count,
                              int pass,
                              bool trust_prior)
{
	enum correspondence_search_flags flags = pass == 0 ? CS_FLAG_SHALLOW_SEARCH : CS_FLAG_DEEP_SEARCH;
	if (trust_prior && dev_state->prior_tilt_trusted) {
		flags |= CS_FLAG_HAVE_POSE_PRIOR | CS_FLAG_TRUST_PRIOR_ORIENT | CS_FLAG_RETURN_BEST_PARTIAL;
	}
	if (search_blob_count <= ASSOC_MATCH_ALL_BLOBS_MAX) {
		flags |= CS_FLAG_MATCH_ALL_BLOBS;
	}
	return flags;
}

/* The shallow->deep fall-through predicate. It keys ONLY on this task's OWN shallow results —
 * never on the shared per-device hypothesis list, which sibling scopes merge into in pool
 * completion order — so the work spend is a pure function of the frame. A GOOD task-local
 * result is the stand-in for "usable": a partial-only shallow result may still become usable
 * after the multi-view merge, so it conservatively keeps the deep pass (allowance-bounded
 * either way). NOTE this skip is BEHAVIOR-CHANGING, not free: in the clutter capture's 1426
 * both-succeed scopes the deep result differed from the shallow one in 99.9% (deep had better
 * reproj-per-match in 52.9%) — the surviving hypothesis changes on those frames and the
 * validation battery adjudicates the decision-shape movement. */
static bool
association_cold_scope_found_good(const struct association_cold_scope_task *task)
{
	/* STRONG, not GOOD: a mirror flip can pass GOOD with plausible reprojection but essentially
	 * never STRONG (<0.75px at 6+ LEDs). Skipping the deep pass on a merely-GOOD shallow result
	 * let one gross flip (135 deg tilt) through the hand-GT scorecard; STRONG-only keeps the
	 * deep pass wherever the shallow result is still flip-ambiguous. */
	for (int i = 0; i < task->n_results[0]; i++) {
		if (POSE_HAS_FLAGS(&task->results[0][i].score, POSE_MATCH_STRONG)) {
			return true;
		}
	}
	return false;
}

static void
association_run_cold_scope_task(void *ptr)
{
	struct association_cold_scope_task *task = ptr;
	correspondence_search_set_blobs(task->cs, task->blobs, task->num_blobs);
	for (int pass = 0; pass < 2; pass++) {
		if (pass == 1 && (!task->run_deep || association_cold_scope_found_good(task))) {
			break;
		}
		struct xrt_pose P_cam_obj = task->P_cam_obj;
		task->n_results[pass] = correspondence_search_find_pose_candidates(
		    task->cs, task->search_model, task->pass_flags[pass],
		    task->work_allowance - task->work_spent, &P_cam_obj, &task->prior_pos_error,
		    &task->prior_rot_error, &task->cam_gravity_vector, task->prior_tilt_sigma_rad,
		    task->prior_yaw_sigma_rad, (float)FLIP_COST_HUBER_KNEE_SIGMA, (float)FLIP_COST_WEIGHT,
		    task->results[pass], CORRESPONDENCE_SEARCH_MAX_RESULTS);
		correspondence_search_get_last_diagnostics(task->cs, &task->diag[pass]);
		task->work_spent += task->diag[pass].work_spent;
		task->pass_ran[pass] = true;
	}
}

static void
association_merge_cold_scope_task(struct t_constellation_tracker *ct,
                                  struct association_device_work *work,
                                  struct tracking_sample_device_state *dev_state,
                                  struct constellation_tracking_sample *sample,
                                  const struct association_cold_scope_task *task)
{
	for (int pass = 0; pass < 2; pass++) {
		if (!task->pass_ran[pass]) {
			continue;
		}
		telem_emit_search_result(task->device_id, task->view_id, task->timestamp_ns,
		                         task->pass_base + pass, task->pass_flags[pass],
		                         task->n_results[pass] > 0, task->prior_tilt_trusted,
		                         &task->diag[pass]);
		for (int i = 0; i < task->n_results[pass]; i++) {
			uint16_t flags = task->ignore_prior_results ? ASSOC_HYP_IGNORE_PRIOR : ASSOC_HYP_NONE;
			if (!POSE_HAS_FLAGS(&task->results[pass][i].score, POSE_MATCH_GOOD)) {
				flags |= ASSOC_HYP_PARTIAL_ONLY;
			}
			association_add_pose_hypothesis(ct, work, dev_state, sample, task->view_id,
			                                ASSOC_SOURCE_COLD_SEARCH, flags,
			                                &task->results[pass][i].pose, true);
		}
	}
}

static void
association_init_cold_scope_task(struct t_constellation_tracker *ct,
                                 struct association_cold_scope_task *task,
                                 struct tracking_sample_device_state *dev_state,
                                 struct constellation_tracking_sample *sample,
                                 int dev_slot,
                                 int view_id,
                                 struct blob *blobs,
                                 int num_blobs,
                                 bool run_deep,
                                 bool trust_prior,
                                 int pass_base)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;

	task->cs = cam->cold_search[dev_state->dev_index];
	task->search_model = device->search_led_model;
	task->blobs = blobs;
	task->num_blobs = num_blobs;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &task->P_cam_obj);
	task->prior_pos_error = dev_state->prior_pos_error;
	task->prior_rot_error = dev_state->prior_rot_error;
	task->cam_gravity_vector = view->cam_gravity_vector;
	task->prior_yaw_sigma_rad = dev_state->prior_yaw_sigma_rad;
	task->prior_tilt_sigma_rad = dev_state->prior_tilt_sigma_rad;
	task->pass_flags[0] = association_cold_search_flags(dev_state, num_blobs, 0, trust_prior);
	task->pass_flags[1] = association_cold_search_flags(dev_state, num_blobs, 1, trust_prior);
	task->work_allowance = 0;
	task->run_deep = run_deep;
	task->timestamp_ns = sample->timestamp;
	task->device_id = telem_device_id(xdev);
	task->dev_slot = dev_slot;
	task->view_id = view_id;
	task->pass_base = pass_base;
	task->prior_tilt_trusted = dev_state->prior_tilt_trusted;
	task->ignore_prior_results = !trust_prior;
	task->pass_ran[0] = task->pass_ran[1] = false;
	task->n_results[0] = task->n_results[1] = 0;
	task->work_spent = 0;
}

static void
association_dispatch_cold_scopes(struct t_constellation_tracker *ct, int n_tasks)
{
	if (ct->cold_search_group != NULL) {
		for (int t = 0; t < n_tasks; t++) {
			u_worker_group_push(ct->cold_search_group, association_run_cold_scope_task,
			                    &ct->cold_scope_tasks[t]);
		}
		u_worker_group_wait_all(ct->cold_search_group);
		return;
	}
	for (int t = 0; t < n_tasks; t++) {
		association_run_cold_scope_task(&ct->cold_scope_tasks[t]);
	}
}

static bool
association_work_has_lockable_hypothesis(const struct association_device_work *work)
{
	for (int i = 0; work != NULL && i < work->count; i++) {
		if (association_lock_eligible(&work->hyps[i])) {
			return true;
		}
	}
	return false;
}

/* The deterministic cold-search waves. ASSOC_FRAME_WORK_BUDGET is partitioned BEFORE any task
 * runs — per-scope allowances are pure functions of frame content + tracker state (never a
 * shared pool drained in completion order) — and merging back into the shared hypothesis lists
 * happens after each wave's wait_all in fixed (device, view) order. Returns the units spent. */
static uint64_t
association_run_cold_waves(struct t_constellation_tracker *ct,
                           struct association_device_work *work,
                           struct constellation_tracking_sample *sample)
{
	uint64_t total_spent = 0;

	bool is_cold[CONSTELLATION_MAX_DEVICES] = {false};
	bool view_gated[CONSTELLATION_MAX_DEVICES][CONSTELLATION_MAX_CAMERAS] = {{false}};
	int n_cold = 0;
	for (int i = 0; i < sample->n_devices; i++) {
		is_cold[i] = !association_work_has_lockable_hypothesis(&work[i]);
		n_cold += is_cold[i] ? 1 : 0;
	}
	const uint32_t dev_budget = n_cold > 0 ? ASSOC_FRAME_WORK_BUDGET / (uint32_t)n_cold : 0;
	uint32_t dev_spent[CONSTELLATION_MAX_DEVICES] = {0};

	/* Wave B: per cold device, prior-ROI scopes share HALF the device budget, split across the
	 * scope views proportionally to the prior-projected-LED count (largest-remainder rounding,
	 * ties to the lower view index — g2_work_budget_split); the other half is reserved for the
	 * wave-C full-frame remainder. Each scope task runs shallow then conditional deep. */
	int n_tasks = 0;
	for (int i = 0; i < sample->n_devices; i++) {
		if (!is_cold[i]) {
			continue;
		}
		struct tracking_sample_device_state *dev_state = &sample->devices[i];
		struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
		const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;

		struct association_cold_scope_task *scopes[CONSTELLATION_MAX_CAMERAS];
		uint32_t weights[CONSTELLATION_MAX_CAMERAS];
		int n_scopes = 0;
		for (int view_id = 0; view_id < sample->n_views; view_id++) {
			struct tracking_sample_frame *view = sample->views + view_id;
			struct constellation_tracker_camera_state *cam = ct->cam + view_id;
			if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
				continue;
			}
			if (dev_state->prior_tilt_trusted && !dev_state->prior_optical_stale &&
			    association_count_prior_visible_leds_in_view(dev_state, view, cam) < 3) {
				view_gated[i][view_id] = true;
				telem_emit_search_skip(telem_device_id(xdev), view_id, sample->timestamp, 0,
				                       G2_SEARCH_ROI_SKIP_VISIBLE_LT3, CS_FLAG_HAVE_POSE_PRIOR,
				                       dev_state->prior_tilt_trusted,
				                       (uint32_t)view->bwobs->num_blobs, 0, 0);
				continue;
			}
			struct association_cold_scope_task *task = &ct->cold_scope_tasks[n_tasks];
			const struct association_prior_roi_result roi =
			    association_build_prior_roi_blobs(dev_state, view, cam, task->roi_blobs);
			if (roi.status != G2_SEARCH_SUCCESS) {
				enum correspondence_search_flags flags = CS_FLAG_HAVE_POSE_PRIOR;
				if (dev_state->prior_tilt_trusted) {
					flags |= CS_FLAG_TRUST_PRIOR_ORIENT;
				}
				telem_emit_search_skip(telem_device_id(xdev), view_id, sample->timestamp, 0,
				                       roi.status, flags, dev_state->prior_tilt_trusted,
				                       (uint32_t)view->bwobs->num_blobs,
				                       (uint32_t)roi.visible_leds, (uint32_t)roi.blob_count);
				continue;
			}
			association_init_cold_scope_task(ct, task, dev_state, sample, i, view_id,
			                                 task->roi_blobs, roi.blob_count, true, true, 0);
			weights[n_scopes] = (uint32_t)roi.visible_leds;
			scopes[n_scopes++] = task;
			n_tasks++;
		}
		if (n_scopes > 0) {
			uint32_t alloc[CONSTELLATION_MAX_CAMERAS];
			g2_work_budget_split(dev_budget / 2, weights, n_scopes, alloc);
			for (int k = 0; k < n_scopes; k++) {
				scopes[k]->work_allowance = alloc[k];
			}
		}
	}
	association_dispatch_cold_scopes(ct, n_tasks);
	for (int t = 0; t < n_tasks; t++) {
		const struct association_cold_scope_task *task = &ct->cold_scope_tasks[t];
		association_merge_cold_scope_task(ct, &work[task->dev_slot],
		                                  &sample->devices[task->dev_slot], sample, task);
		dev_spent[task->dev_slot] += task->work_spent;
		total_spent += task->work_spent;
	}

	/* Wave C: cold devices still without a usable hypothesis run full-frame scopes with the
	 * device's deterministic remainder (the wave-B spend is input-determined, so the remainder
	 * is too). One ROTATING qualifying view per device (cold_deep_view_rr) owns the deep pass
	 * at ASSOC_COLD_DEEP_VIEW_WEIGHT x a shallow view's split share, so a lost device reaches
	 * full depth in every view within <= n_views frames instead of every device paying for full
	 * depth everywhere each frame (100% of the capture's winning poses were shallow-reachable
	 * at blob_depth <= 3). */
	n_tasks = 0;
	for (int i = 0; i < sample->n_devices; i++) {
		if (!is_cold[i] || association_work_has_strong_hypothesis(&work[i])) {
			continue;
		}
		struct tracking_sample_device_state *dev_state = &sample->devices[i];
		struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

		/* Qualifying views, ordered by prior-projected-LED count descending (insertion sort,
		 * stable: ties keep the lower view index first). The order fixes both which view the
		 * deep slot lands on and who wins the split's remainder units. */
		int view_order[CONSTELLATION_MAX_CAMERAS];
		int view_leds[CONSTELLATION_MAX_CAMERAS];
		int n_q = 0;
		for (int view_id = 0; view_id < sample->n_views; view_id++) {
			struct tracking_sample_frame *view = sample->views + view_id;
			if (view->bwobs == NULL || view->bwobs->num_blobs == 0 || view_gated[i][view_id]) {
				continue;
			}
			const int leds =
			    association_count_prior_visible_leds_in_view(dev_state, view, ct->cam + view_id);
			int k = n_q;
			while (k > 0 && view_leds[k - 1] < leds) {
				view_order[k] = view_order[k - 1];
				view_leds[k] = view_leds[k - 1];
				k--;
			}
			view_order[k] = view_id;
			view_leds[k] = leds;
			n_q++;
		}
		if (n_q == 0) {
			continue;
		}
		const int deep_idx = (int)(device->cold_deep_view_rr % (uint32_t)n_q);
		device->cold_deep_view_rr++;

		uint32_t weights[CONSTELLATION_MAX_CAMERAS];
		for (int k = 0; k < n_q; k++) {
			weights[k] = k == deep_idx ? ASSOC_COLD_DEEP_VIEW_WEIGHT : 1u;
		}
		uint32_t alloc[CONSTELLATION_MAX_CAMERAS];
		g2_work_budget_split(dev_budget - dev_spent[i], weights, n_q, alloc);
		for (int k = 0; k < n_q; k++) {
			struct tracking_sample_frame *view = sample->views + view_order[k];
			struct association_cold_scope_task *task = &ct->cold_scope_tasks[n_tasks++];
			const int n_blobs = association_stage_search_blobs(view, task->roi_blobs);
			association_init_cold_scope_task(ct, task, dev_state, sample, i, view_order[k],
			                                 task->roi_blobs, n_blobs, k == deep_idx, true, 2);
			task->work_allowance = alloc[k];
		}
	}
	association_dispatch_cold_scopes(ct, n_tasks);
	for (int t = 0; t < n_tasks; t++) {
		const struct association_cold_scope_task *task = &ct->cold_scope_tasks[t];
		association_merge_cold_scope_task(ct, &work[task->dev_slot],
		                                  &sample->devices[task->dev_slot], sample, task);
		total_spent += task->work_spent;
	}

	/* Wave D: uncovered-view pickup for devices that already hold a lockable hypothesis. Each
	 * uncovered view takes a FIXED bounded slice of whatever frame budget remains, assigned in
	 * (device, view) order — keeping the old BOUNDED_SEARCH protection (today's pickup spends
	 * p50 512 trials) without being unbounded on good frames or silently starved on bad ones.
	 * Partner-labelled blobs still neither qualify a view as uncovered nor buy it search work
	 * (the measured 64.6M-trial hands-close regression). */
	uint32_t frame_remaining = ASSOC_FRAME_WORK_BUDGET - (uint32_t)total_spent;
	bool resort[CONSTELLATION_MAX_DEVICES] = {false};
	n_tasks = 0;
	for (int i = 0; i < sample->n_devices; i++) {
		if (is_cold[i]) {
			continue;
		}
		struct tracking_sample_device_state *dev_state = &sample->devices[i];
		const int model_id = (ct->devices + dev_state->dev_index)->led_model.id;
		if (!association_work_has_uncovered_blob_view(&work[i], sample, model_id)) {
			continue;
		}
		for (int view_id = 0; view_id < sample->n_views; view_id++) {
			struct tracking_sample_frame *view = sample->views + view_id;
			if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
				continue;
			}
			if (association_view_unclaimed_blob_count(view, model_id) < 4 ||
			    association_work_has_view_hypothesis(&work[i], view_id)) {
				continue;
			}
			struct association_cold_scope_task *task = &ct->cold_scope_tasks[n_tasks++];
			const int n_blobs = association_stage_search_blobs(view, task->roi_blobs);
			association_init_cold_scope_task(ct, task, dev_state, sample, i, view_id,
			                                 task->roi_blobs, n_blobs, true, false, 4);
			task->work_allowance =
			    g2_work_budget_take(&frame_remaining, ASSOC_COLD_UNCOVERED_VIEW_ALLOWANCE);
			resort[i] = true;
		}
	}
	association_dispatch_cold_scopes(ct, n_tasks);
	for (int t = 0; t < n_tasks; t++) {
		const struct association_cold_scope_task *task = &ct->cold_scope_tasks[t];
		association_merge_cold_scope_task(ct, &work[task->dev_slot],
		                                  &sample->devices[task->dev_slot], sample, task);
		total_spent += task->work_spent;
	}
	for (int i = 0; i < sample->n_devices; i++) {
		if (resort[i]) {
			association_resort_work(&work[i]);
		}
	}

	return total_spent;
}

static void
association_build_device_work(struct t_constellation_tracker *ct,
                              struct association_device_work *work,
                              struct tracking_sample_device_state *dev_state,
                              struct constellation_tracking_sample *sample)
{
	*work = (struct association_device_work){0};
	/* Wave A — the warm sources. Never charged against the work budget: a tracked device's
	 * fast path keeps full quality regardless of clutter load. */
	association_add_prior_pose_sources(ct, work, dev_state, sample);
	association_add_joint_pnp_source(ct, work, dev_state, sample);
	association_add_label_sources(ct, work, dev_state, sample);
}

static bool
association_is_compatible_with_chosen(const struct association_pose_hypothesis *candidate,
                                      const struct association_pose_hypothesis *const chosen[],
                                      int chosen_count)
{
	if (candidate == NULL) {
		return true;
	}
	for (int i = 0; i < chosen_count; i++) {
		if (chosen[i] != NULL && !association_joint_pair_is_compatible(candidate, chosen[i])) {
			return false;
		}
	}
	return true;
}

static const struct association_pose_hypothesis *
association_best_lockable_local(const struct association_device_work *work,
                                const struct association_pose_hypothesis *const chosen[],
                                int chosen_count)
{
	for (int h = 0; work != NULL && h < work->count; h++) {
		const struct association_pose_hypothesis *hyp = &work->hyps[h];
		if (association_lock_eligible(hyp) &&
		    association_is_compatible_with_chosen(hyp, chosen, chosen_count)) {
			return hyp;
		}
	}
	return NULL;
}

static uint8_t
association_count_choice_conflicts(const struct association_pose_hypothesis *candidate,
                                   const struct association_pose_hypothesis *const chosen[],
                                   int n_devices,
                                   int own_slot)
{
	uint8_t count = 0;
	for (int i = 0; candidate != NULL && i < n_devices; i++) {
		if (i == own_slot || chosen[i] == NULL) {
			continue;
		}
		count += association_hypotheses_shared_blob_count(candidate, chosen[i]);
	}
	return count;
}

static float
association_device_lower_bound(const struct association_device_work *work)
{
	float best = work->has_blobs ? ASSOC_ABSENT_WITH_BLOBS_COST : 0.0f;
	for (int h = 0; h < work->count; h++) {
		const struct association_pose_hypothesis *candidate = &work->hyps[h];
		if (association_lock_eligible(candidate)) {
			const float cost = association_option_cost(candidate, ASSOC_OBS_POSE_LOCK);
			if (cost < best) {
				best = cost;
			}
		}
		if (association_position_only_eligible(candidate)) {
			const float cost = association_option_cost(candidate, ASSOC_OBS_POSITION_ONLY);
			if (cost < best) {
				best = cost;
			}
		}
		if (association_led_fold_eligible(candidate)) {
			const float cost = association_option_cost(candidate, ASSOC_OBS_LED_FOLD);
			if (cost < best) {
				best = cost;
			}
		}
	}
	return best;
}

static float
association_remaining_lower_bound(const struct association_device_work work[], int n_devices, int index)
{
	float lower_bound = 0.0f;
	for (int i = index; i < n_devices; i++) {
		lower_bound += association_device_lower_bound(&work[i]);
	}
	return lower_bound;
}

static int
association_count_visual_observations(const enum association_observation_kind kind[], int n_devices)
{
	int count = 0;
	for (int i = 0; i < n_devices; i++) {
		if (kind[i] != ASSOC_OBS_ABSENT) {
			count++;
		}
	}
	return count;
}

static bool
association_observation_kind_visual(enum association_observation_kind kind)
{
	return kind == ASSOC_OBS_POSE_LOCK || kind == ASSOC_OBS_POSITION_ONLY || kind == ASSOC_OBS_LED_FOLD;
}

static bool
association_observation_kind_owns_position(enum association_observation_kind kind)
{
	return kind == ASSOC_OBS_POSE_LOCK || kind == ASSOC_OBS_POSITION_ONLY;
}

static float
association_prior_distance_m(const struct association_pose_hypothesis *candidate,
                             const struct tracking_sample_device_state *dev_state)
{
	if (candidate == NULL || dev_state == NULL || !dev_state->prior_tilt_trusted ||
	    (candidate->flags & ASSOC_HYP_HAS_POSE) == 0) {
		return INFINITY;
	}
	return m_vec3_len(m_vec3_sub(candidate->pose_world.position, dev_state->P_world_obj_prior.position));
}

/* Has device @p j's chosen observation actually VACATED the identity device @p i is being charged for
 * stealing? Holding a position of its own excuses the contest only if the partner went somewhere that
 * is not device @p i's own prior. A mutual transposition — each device sitting on the other's prior —
 * is self-consistent and would otherwise buy BOTH sides the excuse, which is precisely the assignment
 * this term exists to reject: on xv1/periodic-300 t=60.918 it cut a 52.3-nat steal to 18.3, under the
 * 28.0 nats of coasting, so the swapped pair won outright and both controllers stayed transposed for
 * ~1 s. */
static bool
association_identity_partner_vacated(const struct constellation_tracking_sample *sample,
                                     const struct association_pose_hypothesis *const chosen[],
                                     const enum association_observation_kind chosen_kind[],
                                     int i,
                                     int j)
{
	if (chosen[j] == NULL || !association_observation_kind_owns_position(chosen_kind[j])) {
		return false;
	}
	const float partner_own_d = association_prior_distance_m(chosen[j], &sample->devices[j]);
	return association_identity_steal_excess_m(partner_own_d,
	                                           association_prior_distance_m(chosen[j],
	                                                                        &sample->devices[i])) <= 0.0f;
}

static float
association_cross_device_identity_nll(const struct constellation_tracking_sample *sample,
                                      const struct association_pose_hypothesis *const chosen[],
                                      const enum association_observation_kind chosen_kind[],
                                      int n_devices)
{
	float total = 0.0f;

	for (int i = 0; i < n_devices; i++) {
		const struct association_pose_hypothesis *candidate = chosen[i];
		if (candidate == NULL || !association_observation_kind_visual(chosen_kind[i])) {
			continue;
		}

		const float own_d = association_prior_distance_m(candidate, &sample->devices[i]);
		if (!isfinite(own_d)) {
			continue;
		}

		for (int j = 0; j < n_devices; j++) {
			if (i == j) {
				continue;
			}

			const float other_d = association_prior_distance_m(candidate, &sample->devices[j]);
			const float steal_m = association_identity_steal_excess_m(own_d, other_d);
			if (steal_m <= 0.0f) {
				continue;
			}

			const float s = steal_m / ASSOC_IDENTITY_STEAL_SIGMA_M;
			float nll = ASSOC_IDENTITY_STEAL_BASE_NLL + 0.5f * s * s;
			if (association_identity_partner_vacated(sample, chosen, chosen_kind, i, j)) {
				nll *= ASSOC_IDENTITY_STEAL_VACATED_SCALE;
			}
			if (nll > ASSOC_IDENTITY_STEAL_MAX_NLL) {
				nll = ASSOC_IDENTITY_STEAL_MAX_NLL;
			}
			total += nll;
		}
	}

	return total;
}

static void
association_emit_lockable_not_chosen(struct t_constellation_tracker *ct,
                                     const struct tracking_sample_device_state *dev_state,
                                     const struct constellation_tracking_sample *sample,
                                     const struct association_pose_hypothesis *hyp,
                                     const struct association_pose_hypothesis *const chosen[],
                                     int slot)
{
	if (!g2_telem_enabled() || hyp == NULL) {
		return;
	}
	const struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;
	const uint8_t conflicts = association_count_choice_conflicts(hyp, chosen, sample->n_devices, slot);
	g2_telem_event(telem_device_id(xdev), sample->timestamp, G2_TELEM_EV_ASSOC_LOCKABLE_NOT_CHOSEN,
	               (float)conflicts);
}

struct joint_contention_row
{
	int device_slot;
	int led_id;
	bool valid[CONSTELLATION_MAX_CAMERAS];
	struct xrt_vec2 pos_px[CONSTELLATION_MAX_CAMERAS];
	double gate_ax_px[CONSTELLATION_MAX_CAMERAS];
	double gate_ay_px[CONSTELLATION_MAX_CAMERAS];
	double led_radius_px[CONSTELLATION_MAX_CAMERAS];
	double p_det;
	double rho;
};

struct joint_contention_blob
{
	int view_id;
	int blob_id;
};

struct joint_contention_device
{
	struct association_pose_hypothesis *rep;
	bool contended;
};

static void
joint_contention_fill_view(struct t_constellation_tracker *ct,
                           struct constellation_tracking_sample *sample,
                           int device_slot,
                           const struct association_pose_hypothesis *rep,
                           int view_id,
                           struct joint_contention_row *rows,
                           int n_rows)
{
	struct tracking_sample_frame *view = sample->views + view_id;
	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return;
	}

	struct tracking_sample_device_state *dev_state = sample->devices + device_slot;
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;

	struct xrt_pose P_cam_obj;
	math_pose_transform(&view->P_cam_world, &rep->pose_world, &P_cam_obj);

	struct pose_metrics_blob_match_info match_info;
	pose_metrics_match_pose_to_blobs(&P_cam_obj, view->bwobs->blobs, view->bwobs->num_blobs, &device->led_model,
	                                 &cam->camera_model, &match_info);

	for (int r = 0; r < n_rows; r++) {
		if (rows[r].device_slot != device_slot) {
			continue;
		}
		for (int i = 0; i < match_info.num_visible_leds; i++) {
			const struct pose_metrics_visible_led_info *visible = &match_info.visible_leds[i];
			if (visible->led == NULL || visible->led->id != rows[r].led_id) {
				continue;
			}
			rows[r].valid[view_id] = true;
			rows[r].pos_px[view_id] = visible->pos_px;
			rows[r].gate_ax_px[view_id] = visible->gate_ax_px;
			rows[r].gate_ay_px[view_id] = visible->gate_ay_px;
			rows[r].led_radius_px[view_id] = visible->led_radius_px;
			const double p = pose_metrics_pkf_detection_prob(visible->facing_dot);
			if (p > rows[r].p_det) {
				rows[r].p_det = p;
			}
			break;
		}
	}
}

static double
joint_contention_entry(const struct joint_contention_row *row,
                       const struct joint_contention_blob *blob_ref,
                       struct constellation_tracking_sample *sample)
{
	const int view_id = blob_ref->view_id;
	if (view_id < 0 || view_id >= sample->n_views || !row->valid[view_id]) {
		return 0.0;
	}
	struct tracking_sample_frame *view = sample->views + view_id;
	if (view->bwobs == NULL || blob_ref->blob_id < 0 || blob_ref->blob_id >= view->bwobs->num_blobs) {
		return 0.0;
	}
	const struct blob *b = &view->bwobs->blobs[blob_ref->blob_id];
	if (b->width > row->led_radius_px[view_id] * 4.0 || b->height > row->led_radius_px[view_id] * 4.0) {
		return 0.0;
	}
	const double dx = row->pos_px[view_id].x - b->x;
	const double dy = row->pos_px[view_id].y - b->y;
	const double ax = row->gate_ax_px[view_id];
	const double ay = row->gate_ay_px[view_id];
	if (!(ax > 0.0) || !(ay > 0.0) || (dx * dx) / (ax * ax) + (dy * dy) / (ay * ay) > 1.0) {
		return 0.0;
	}
	return exp(-pose_metrics_pkf_pair_nll(dx * dx + dy * dy, row->p_det));
}

static void
association_resort_work(struct association_device_work *work)
{
	for (int i = 1; work != NULL && i < work->count; i++) {
		struct association_pose_hypothesis cur = work->hyps[i];
		int j = i;
		while (j > 0 && association_hypothesis_less(&cur, &work->hyps[j - 1])) {
			work->hyps[j] = work->hyps[j - 1];
			j--;
		}
		work->hyps[j] = cur;
	}
}

static bool
joint_contention_blob_seen(const struct joint_contention_blob *blobs, int n_blobs, const struct association_blob_ref *ref)
{
	for (int i = 0; i < n_blobs; i++) {
		if (blobs[i].view_id == ref->view_id && blobs[i].blob_id == ref->blob_id) {
			return true;
		}
	}
	return false;
}

static void
association_apply_joint_contention(struct t_constellation_tracker *ct,
                                   struct association_device_work work[],
                                   struct constellation_tracking_sample *sample)
{
	if (sample->n_devices < 2) {
		return;
	}

	struct joint_contention_device dev[CONSTELLATION_MAX_DEVICES] = {0};
	int n_reps = 0;
	for (int s = 0; s < sample->n_devices; s++) {
		const struct association_pose_hypothesis *best_const =
		    association_best_lockable_local(&work[s], NULL, 0);
		if (best_const != NULL) {
			dev[s].rep = (struct association_pose_hypothesis *)best_const;
			n_reps++;
		}
	}
	if (n_reps < 2) {
		return;
	}

	struct joint_contention_blob blobs[ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS] = {0};
	int n_blobs = 0;
	for (int a = 0; a < sample->n_devices; a++) {
		if (dev[a].rep == NULL) {
			continue;
		}
		for (int b = a + 1; b < sample->n_devices; b++) {
			if (dev[b].rep == NULL) {
				continue;
			}
			for (uint8_t ai = 0; ai < dev[a].rep->matched_count; ai++) {
				const struct association_blob_ref *ra = &dev[a].rep->matched_blobs[ai];
				if (!association_blob_ref_is_valid(ra)) {
					continue;
				}
				bool shared = false;
				for (uint8_t bi = 0; bi < dev[b].rep->matched_count; bi++) {
					if (association_blob_ref_equal(ra, &dev[b].rep->matched_blobs[bi])) {
						shared = true;
						break;
					}
				}
				if (!shared) {
					continue;
				}
				dev[a].contended = true;
				dev[b].contended = true;
				if (!joint_contention_blob_seen(blobs, n_blobs, ra) && n_blobs < (int)ARRAY_SIZE(blobs)) {
					blobs[n_blobs++] = (struct joint_contention_blob){ra->view_id, ra->blob_id};
				}
			}
		}
	}
	if (n_blobs == 0 || n_blobs > PKF_MAX_CLUSTER) {
		return;
	}

	struct joint_contention_row rows[PKF_MAX_CLUSTER] = {0};
	int n_rows = 0;
	for (int s = 0; s < sample->n_devices; s++) {
		if (!dev[s].contended || dev[s].rep == NULL) {
			continue;
		}
		const struct association_pose_hypothesis *rep = dev[s].rep;
		const double per_led_rho = rep->matched_count > 0
		                               ? rep->cost.reprojection_nll
		                               : 0.0;
		for (uint8_t mi = 0; mi < rep->matched_count; mi++) {
			const struct association_blob_ref *ref = &rep->matched_blobs[mi];
			if (!joint_contention_blob_seen(blobs, n_blobs, ref)) {
				continue;
			}
			bool duplicate = false;
			for (int r = 0; r < n_rows; r++) {
				if (rows[r].device_slot == s && rows[r].led_id == rep->matched_led_ids[mi]) {
					duplicate = true;
					break;
				}
			}
			if (duplicate) {
				continue;
			}
			if (n_rows >= PKF_MAX_CLUSTER) {
				return;
			}
			rows[n_rows].device_slot = s;
			rows[n_rows].led_id = rep->matched_led_ids[mi];
			rows[n_rows].rho = per_led_rho;
			n_rows++;
		}
	}
	if (n_rows < 2) {
		return;
	}

	for (int s = 0; s < sample->n_devices; s++) {
		if (!dev[s].contended || dev[s].rep == NULL) {
			continue;
		}
		for (int v = 0; v < sample->n_views; v++) {
			joint_contention_fill_view(ct, sample, s, dev[s].rep, v, rows, n_rows);
		}
	}

	struct joint_contention_cluster cluster = {0};
	cluster.n_rows = n_rows;
	cluster.n_cols = n_blobs;
	const double clutter = pose_metrics_pkf_clutter_likelihood();
	for (int r = 0; r < n_rows; r++) {
		cluster.row_device[r] = rows[r].device_slot;
		cluster.row_rho[r] = rows[r].rho;
		cluster.row_logodds[r] =
		    rows[r].p_det > 0.0 ? (-log(rows[r].p_det) + log(1.0 - rows[r].p_det)) : 0.0;
		cluster.L_clutter[r] = clutter;
		for (int j = 0; j < n_blobs; j++) {
			cluster.L[r * n_blobs + j] = joint_contention_entry(&rows[r], &blobs[j], sample);
		}
	}

	const double ambig_gap = -log(PKF_TAU_AMBIG);
	bool ambiguous = false;
	for (int j = 0; j < n_blobs && !ambiguous; j++) {
		double best = INFINITY;
		double second = INFINITY;
		int best_dev = -1;
		bool cross_device = false;
		for (int r = 0; r < n_rows; r++) {
			const double l = cluster.L[r * n_blobs + j];
			if (!(l > 0.0)) {
				continue;
			}
			const double nll = -log(l);
			const int dev_slot = rows[r].device_slot;
			if (nll < best) {
				if (best_dev >= 0 && best_dev != dev_slot) {
					cross_device = true;
				}
				second = best;
				best = nll;
				best_dev = dev_slot;
			} else if (nll < second) {
				second = nll;
				if (best_dev != dev_slot) {
					cross_device = true;
				}
			}
		}
		ambiguous = cross_device && isfinite(second) && (second - best) < ambig_gap;
	}
	if (!ambiguous) {
		return;
	}

	const struct joint_contention_result split = joint_contention_split(&cluster, sample->n_devices - 1);
	if (!split.valid) {
		return;
	}

	bool changed = false;
	for (int s = 0; s < sample->n_devices; s++) {
		if (!dev[s].contended || dev[s].rep == NULL || !(split.dev_hard_rho[s] > 0.0)) {
			continue;
		}
		const double delta_sum = split.dev_marginal[s] - split.dev_hard_rho[s];
		if (!(delta_sum < 0.0)) {
			continue;
		}
		const double per_match_delta = delta_sum / (double)(dev[s].rep->matched_count > 0 ? dev[s].rep->matched_count : 1);
		dev[s].rep->cost.joint_contention_delta_nll += (float)per_match_delta;
		changed = true;
		if (g2_telem_enabled()) {
			struct tracking_sample_device_state *dev_state = sample->devices + s;
			const struct constellation_tracker_device *cdev = ct->devices + dev_state->dev_index;
			const struct xrt_device *xdev = cdev->connection != NULL ? cdev->connection->xdev : NULL;
			g2_telem_event(telem_device_id(xdev), sample->timestamp, G2_TELEM_EV_ASSOC_JOINT_CONTENTION,
			               (float)n_blobs);
		}
	}
	if (changed) {
		for (int s = 0; s < sample->n_devices; s++) {
			association_resort_work(&work[s]);
		}
	}
}

/* @p banned_slot / @p banned_hyp (optional, -1 / NULL to disable): exclude every hypothesis of
 * @p banned_slot that touches ANY blob of @p banned_hyp — the "this cluster is NOT this device's"
 * counterfactual the identity-ambiguity margin is measured against. Touching one blob is enough to
 * ban: a clean alternative world must not smuggle any of the contested blobs back to the device
 * whose claim is being withdrawn. */
static void
association_select_joint_recursive(const struct association_device_work work[],
                                   const struct constellation_tracking_sample *sample,
                                   int n_devices,
                                   int index,
                                   const struct association_pose_hypothesis *chosen[],
                                   enum association_observation_kind chosen_kind[],
                                   float cost,
                                   int banned_slot,
                                   const struct association_pose_hypothesis *banned_hyp,
                                   struct association_joint_choice *best)
{
	if (index == n_devices) {
		const int visual_count = association_count_visual_observations(chosen_kind, n_devices);
		const float total_cost = cost + association_cross_device_identity_nll(sample, chosen, chosen_kind, n_devices);
		if (!best->valid || total_cost < best->total_cost - 1e-4f ||
		    (fabsf(total_cost - best->total_cost) <= 1e-4f && visual_count > best->visual_count)) {
			best->valid = true;
			best->total_cost = total_cost;
			best->visual_count = visual_count;
			for (int i = 0; i < n_devices; i++) {
				best->chosen[i] = chosen[i];
				best->kind[i] = chosen_kind[i];
			}
		}
		return;
	}

	const float absent_cost = work[index].has_blobs ? ASSOC_ABSENT_WITH_BLOBS_COST : 0.0f;
	if (best->valid && cost + association_remaining_lower_bound(work, n_devices, index) > best->total_cost) {
		return;
	}
	if (!best->valid ||
	    cost + absent_cost + association_remaining_lower_bound(work, n_devices, index + 1) <= best->total_cost) {
		chosen[index] = NULL;
		chosen_kind[index] = ASSOC_OBS_ABSENT;
		association_select_joint_recursive(work, sample, n_devices, index + 1, chosen, chosen_kind,
		                                   cost + absent_cost, banned_slot, banned_hyp, best);
	}

	for (int h = 0; h < work[index].count; h++) {
		const struct association_pose_hypothesis *candidate = &work[index].hyps[h];
		if (index == banned_slot && association_hypotheses_share_blob(candidate, banned_hyp)) {
			continue;
		}
		const enum association_observation_kind kinds[] = {
		    ASSOC_OBS_POSE_LOCK,
		    ASSOC_OBS_POSITION_ONLY,
		    ASSOC_OBS_LED_FOLD,
		};
		for (size_t k = 0; k < ARRAY_SIZE(kinds); k++) {
			const enum association_observation_kind kind = kinds[k];
			if ((kind == ASSOC_OBS_POSE_LOCK && !association_lock_eligible(candidate)) ||
			    (kind == ASSOC_OBS_POSITION_ONLY && !association_position_only_eligible(candidate)) ||
			    (kind == ASSOC_OBS_LED_FOLD && !association_led_fold_eligible(candidate))) {
				continue;
			}
			const float option_cost = association_option_cost(candidate, kind);
			if (best->valid &&
			    cost + option_cost + association_remaining_lower_bound(work, n_devices, index + 1) >
			        best->total_cost) {
				continue;
			}
			if (!association_is_compatible_with_chosen(candidate, chosen, index)) {
				continue;
			}
			chosen[index] = candidate;
			chosen_kind[index] = kind;
			association_select_joint_recursive(work, sample, n_devices, index + 1, chosen, chosen_kind,
			                                   cost + option_cost, banned_slot, banned_hyp, best);
		}
	}
	chosen[index] = NULL;
	chosen_kind[index] = ASSOC_OBS_ABSENT;
}

static void
association_cache_hypothesis_pnp_pose(struct t_constellation_tracker *ct,
                                      struct tracking_sample_device_state *dev_state,
                                      struct constellation_tracking_sample *sample,
                                      const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->matched_count < 4) {
		return;
	}
	const bool multiview = association_distinct_view_count(hyp) >= 2;
	const bool stale_single_view_recovery = association_stale_prior_visual_recovery_eligible(hyp);
	if (!multiview && !stale_single_view_recovery) {
		return;
	}
	if (association_single_view_prior_disagrees(hyp) && !stale_single_view_recovery) {
		return;
	}
	if (association_reprojection_per_match(hyp) > ASSOC_POSITION_ONLY_MAX_REPROJ_PER_LED ||
	    hyp->cost.total_nll >= ASSOC_VISUAL_AMBIG_COST) {
		return;
	}
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct xrt_pose P_xrworld_model;
	pose_flip_YZ(&hyp->pose_world, &P_xrworld_model);

	struct xrt_pose P_xrworld_device;
	math_pose_transform(&P_xrworld_model, &device->led_model.P_model_device, &P_xrworld_device);
	constellation_tracked_device_connection_cache_pnp_pose_candidate(device->connection, sample->timestamp, dev_state->estimator_prior,
	                                                                &P_xrworld_device);
}

static void
association_fold_ambiguous_hypothesis(struct t_constellation_tracker *ct,
                                      struct tracking_sample_device_state *dev_state,
                                      struct constellation_tracking_sample *sample,
                                      const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->primary_view_id < 0 || hyp->primary_view_id >= sample->n_views) {
		return;
	}
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + hyp->primary_view_id;
	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return;
	}
	association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp, hyp, true, 0,
	                           NULL);
	association_cache_hypothesis_pnp_pose(ct, dev_state, sample, hyp);
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		association_fold_hypothesis_view(ct, dev_state, sample, hyp, view_id, false);
	}
}

static void
association_fold_triangulated_position(struct t_constellation_tracker *ct,
                                       struct tracking_sample_device_state *dev_state,
                                       struct constellation_tracking_sample *sample,
                                       const struct multicam_tri_result *res)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

	struct xrt_pose P_cvworld_model = {.orientation = {0.f, 0.f, 0.f, 1.f}, .position = res->position};
	struct xrt_pose P_xrworld_model;
	pose_flip_YZ(&P_cvworld_model, &P_xrworld_model);

	struct xrt_pose P_xrworld_device;
	math_pose_transform(&P_xrworld_model, &device->led_model.P_model_device, &P_xrworld_device);

	const float std_m = res->position_std_m;
	const struct xrt_vec3 position_variance = {std_m * std_m, std_m * std_m, std_m * std_m};
	constellation_tracked_device_connection_notify_position(device->connection, sample->timestamp, dev_state->estimator_prior,
	                                                        &P_xrworld_device.position,
	                                                        &position_variance, true);
}

static void
association_l1_census_second_view(const int view_leds[],
                                  int committed_view,
                                  int n_views,
                                  int *out_best,
                                  int *out_n_ge_refine)
{
	int best = 0;
	int n_ge = view_leds[committed_view] >= ASSOC_L1_REFINE_PER_CAM_LEDS ? 1 : 0;
	for (int view_id = 0; view_id < n_views; view_id++) {
		if (view_id == committed_view) {
			continue;
		}
		if (view_leds[view_id] > best) {
			best = view_leds[view_id];
		}
		if (view_leds[view_id] >= ASSOC_L1_REFINE_PER_CAM_LEDS) {
			n_ge++;
		}
	}
	*out_best = best;
	*out_n_ge_refine = n_ge;
}

static bool
association_l1_labelled_triangulate(struct t_constellation_tracker *ct,
                                    struct tracking_sample_device_state *dev_state,
                                    struct constellation_tracking_sample *sample,
                                    const struct association_pose_hypothesis *hyp,
                                    int view_led_count[CONSTELLATION_MAX_CAMERAS],
                                    struct multicam_tri_result *out_res)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct blob view_blobs[CONSTELLATION_MAX_CAMERAS][ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS];

	for (uint8_t i = 0; i < hyp->matched_count; i++) {
		const struct association_blob_ref *ref = &hyp->matched_blobs[i];
		if (!association_blob_ref_is_valid(ref) || ref->view_id < 0 || ref->view_id >= sample->n_views ||
		    hyp->matched_led_ids[i] < 0 || hyp->matched_led_ids[i] >= device->led_model.num_leds) {
			continue;
		}
		struct tracking_sample_frame *view = sample->views + ref->view_id;
		if (view->bwobs == NULL || ref->blob_id < 0 || ref->blob_id >= view->bwobs->num_blobs ||
		    view_led_count[ref->view_id] >= ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS) {
			continue;
		}

		struct blob labelled = view->bwobs->blobs[ref->blob_id];
		labelled.led_id = LED_MAKE_ID(device->led_model.id, hyp->matched_led_ids[i]);
		view_blobs[ref->view_id][view_led_count[ref->view_id]++] = labelled;
	}

	struct multicam_tri_view tri_views[CONSTELLATION_MAX_CAMERAS];
	int n_tri_views = 0;
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		if (view_led_count[view_id] <= 0) {
			continue;
		}
		tri_views[n_tri_views].blobs = view_blobs[view_id];
		tri_views[n_tri_views].num_blobs = view_led_count[view_id];
		tri_views[n_tri_views].calib = &ct->cam[view_id].camera_model;
		tri_views[n_tri_views].P_world_cam = sample->views[view_id].P_world_cam;
		n_tri_views++;
	}

	return n_tri_views >= 2 &&
	       multicam_triangulate_position(tri_views, n_tri_views, &device->led_model,
	                                      &hyp->pose_world.orientation, dev_state->prior_yaw_sigma_rad,
	                                      ASSOC_L1_MIN_DISPUTE_LEDS, out_res);
}

static bool
association_l1_epipolar_triangulate(struct t_constellation_tracker *ct,
                                    struct tracking_sample_device_state *dev_state,
                                    struct constellation_tracking_sample *sample,
                                    int committed_view,
                                    int view_led_count[CONSTELLATION_MAX_CAMERAS],
                                    struct multicam_tri_result *out_res)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct multicam_tri_view epi_views[CONSTELLATION_MAX_CAMERAS];
	int epi_sample_view[CONSTELLATION_MAX_CAMERAS];
	int n_epi_views = 0;
	int committed_compact_view = -1;
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs <= 0) {
			continue;
		}
		epi_views[n_epi_views].blobs = view->bwobs->blobs;
		epi_views[n_epi_views].num_blobs = view->bwobs->num_blobs;
		epi_views[n_epi_views].calib = &ct->cam[view_id].camera_model;
		epi_views[n_epi_views].P_world_cam = view->P_world_cam;
		epi_sample_view[n_epi_views] = view_id;
		if (view_id == committed_view) {
			committed_compact_view = n_epi_views;
		}
		n_epi_views++;
	}

	if (n_epi_views < 2 || committed_compact_view < 0) {
		return false;
	}

	if (!dev_state->prior_tilt_trusted) {
		return false;
	}

	const float reach_m = association_epipolar_reach_m(dev_state);

	struct multicam_tri_result epi = {0};
	if (n_epi_views < 2 ||
	    !multicam_triangulate_epipolar_position(epi_views, n_epi_views, &device->led_model,
	                                            &dev_state->P_world_obj_prior,
	                                            dev_state->prior_yaw_sigma_rad, reach_m,
	                                            ASSOC_L1_EPIPOLAR_MODEL_GATE_M,
	                                            ASSOC_L1_MIN_DISPUTE_LEDS, &epi)) {
		return false;
	}

	memset(view_led_count, 0, sizeof(int) * CONSTELLATION_MAX_CAMERAS);
	for (int i = 0; i < epi.num_blob_refs; i++) {
		const int compact_view = epi.blob_refs[i].view_id;
		if (compact_view >= 0 && compact_view < n_epi_views) {
			const int sample_view = epi_sample_view[compact_view];
			if (sample_view >= 0 && sample_view < CONSTELLATION_MAX_CAMERAS) {
				view_led_count[sample_view]++;
			}
		}
	}
	*out_res = epi;
	return true;
}

static enum l1_verdict
association_l1_depth_check(struct t_constellation_tracker *ct,
                           struct tracking_sample_device_state *dev_state,
                           struct constellation_tracking_sample *sample,
                           const struct association_pose_hypothesis *hyp,
                           struct multicam_tri_result *out_res)
{
	const int committed_view = hyp->primary_view_id;
	int view_led_count[CONSTELLATION_MAX_CAMERAS] = {0};
	struct multicam_tri_result res = {0};
	bool tri_ok = association_l1_labelled_triangulate(ct, dev_state, sample, hyp, view_led_count, &res);

	int best_second_view_leds = 0;
	int n_cams_ge_refine = 0;
	association_l1_census_second_view(view_led_count, committed_view, sample->n_views, &best_second_view_leds,
	                                  &n_cams_ge_refine);

	if (!tri_ok &&
	    association_l1_epipolar_triangulate(ct, dev_state, sample, committed_view, view_led_count, &res)) {
		tri_ok = true;
		association_l1_census_second_view(view_led_count, committed_view, sample->n_views,
		                                  &best_second_view_leds, &n_cams_ge_refine);
	}

	const struct xrt_vec3 dp = m_vec3_sub(res.position, hyp->pose_world.position);
	const struct l1_depth_evidence evidence = {
	    .disp_m = m_vec3_len(dp),
	    .triangulated_std_m = (double)res.position_std_m,
	    .committed_std_m = (double)association_position_observation_std_m(hyp),
	    .best_second_view_leds = best_second_view_leds,
	    .n_cams_ge_refine_leds = n_cams_ge_refine,
	    .tri_succeeded = tri_ok,
	};
	const struct l1_depth_params params = {
	    .dispute_sigma = ASSOC_L1_DISPUTE_SIGMA,
	    .min_dispute_leds = ASSOC_L1_MIN_DISPUTE_LEDS,
	    .refine_per_cam_leds = ASSOC_L1_REFINE_PER_CAM_LEDS,
	    .refine_max_std_m = ASSOC_L1_REFINE_MAX_STD_M,
	};

	const enum l1_verdict verdict = l1_depth_verdict(&evidence, &params);
	if (verdict == L1V_REFINE) {
		*out_res = res;
	}
	return verdict;
}

static bool
association_commit_joint_pose(struct t_constellation_tracker *ct,
                              struct tracking_sample_device_state *dev_state,
                              struct constellation_tracking_sample *sample,
                              const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->primary_view_id < 0 || hyp->primary_view_id >= sample->n_views) {
		return false;
	}
	struct tracking_sample_frame *view = sample->views + hyp->primary_view_id;
	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return false;
	}
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	const int selected_matches =
	    association_apply_hypothesis_matches(dev_state, device, view, hyp, hyp->primary_view_id);
	if (selected_matches < 4 && hyp->matched_count < 4) {
		const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;
		g2_telem_event(telem_device_id(xdev), sample->timestamp, G2_TELEM_EV_ASSOC_LOCK_COMMIT_FAILED,
		               (float)selected_matches);
		association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp, hyp,
		                           true, 0, NULL);
		return false;
	}

	if (association_distinct_view_count(hyp) < 2) {
		struct multicam_tri_result tri_res = {0};
		const enum l1_verdict verdict = association_l1_depth_check(ct, dev_state, sample, hyp, &tri_res);
		if (g2_telem_enabled()) {
			const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;
			g2_telem_event(telem_device_id(xdev), sample->timestamp, G2_TELEM_EV_ASSOC_L1_DEPTH_CHECK,
			               (float)verdict);
		}
		if (verdict == L1V_DISPUTE) {
			association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp,
			                           hyp, true, 0, NULL);
			return false;
		}
		if (verdict == L1V_REFINE) {
			association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp,
			                           hyp, true, 0, NULL);
			association_fold_triangulated_position(ct, dev_state, sample, &tri_res);
			return true;
		}
	}

	association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp, hyp, true, 1,
	                           NULL);
	dev_state->score = hyp->score;
	dev_state->score.matched_blobs = hyp->matched_count;
	struct xrt_pose pose = hyp->pose_cam;
	submit_device_pose(ct, dev_state, sample, hyp->primary_view_id, &pose);
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		if (view_id == hyp->primary_view_id) {
			continue;
		}
		const int view_matches = association_fold_hypothesis_view(ct, dev_state, sample, hyp, view_id, true);
		/* This contributing view also observed and folded the committed pose; record its commit with the
		 * pose re-expressed in this camera's frame so the per-camera analysis sees the lock here too (the
		 * lock is one device pose shared across views, not one-per-camera). Same >=4 matched-LED bar as the
		 * primary lock so a view contributing only a stray LED or two isn't logged as a full pose commit. */
		if (view_matches >= 4) {
			struct tracking_sample_frame *fold_view = sample->views + view_id;
			struct xrt_pose P_cam_obj_view;
			math_pose_transform(&fold_view->P_cam_world, &hyp->pose_world, &P_cam_obj_view);
			association_emit_candidate(device, dev_state, fold_view, view_id, sample->timestamp, hyp, true,
			                           1, &P_cam_obj_view);
		}
	}
	return true;
}

static bool
association_commit_position_only(struct t_constellation_tracker *ct,
                                 struct tracking_sample_device_state *dev_state,
                                 struct constellation_tracking_sample *sample,
                                 const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->primary_view_id < 0 || hyp->primary_view_id >= sample->n_views) {
		return false;
	}

	struct tracking_sample_frame *view = sample->views + hyp->primary_view_id;
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct xrt_vec3 position;
	float std_m = association_position_observation_std_m(hyp);
	bool used_pnp_position = false;
	if (POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_POSITION)) {
		struct xrt_pose P_xrworld_model;
		pose_flip_YZ(&hyp->pose_world, &P_xrworld_model);

		struct xrt_pose P_xrworld_device;
		math_pose_transform(&P_xrworld_model, &device->led_model.P_model_device, &P_xrworld_device);
		position = P_xrworld_device.position;
		used_pnp_position = true;
	} else {
		int view_led_count[CONSTELLATION_MAX_CAMERAS] = {0};
		struct multicam_tri_result tri_res = {0};
		bool tri_ok = association_l1_labelled_triangulate(ct, dev_state, sample, hyp, view_led_count, &tri_res);
		if (!tri_ok) {
			tri_ok = association_l1_epipolar_triangulate(ct, dev_state, sample, hyp->primary_view_id,
			                                             view_led_count, &tri_res);
		}
		if (!tri_ok || tri_res.position_std_m > ASSOC_L1_REFINE_MAX_STD_M ||
		    tri_res.num_leds < ASSOC_L1_MIN_DISPUTE_LEDS) {
			return false;
		}
		struct xrt_pose P_cvworld_model = {.orientation = {0.f, 0.f, 0.f, 1.f}, .position = tri_res.position};
		struct xrt_pose P_xrworld_model;
		pose_flip_YZ(&P_cvworld_model, &P_xrworld_model);
		struct xrt_pose P_xrworld_device;
		math_pose_transform(&P_xrworld_model, &device->led_model.P_model_device, &P_xrworld_device);
		position = P_xrworld_device.position;
		std_m = tri_res.position_std_m;
	}
	const struct xrt_vec3 position_variance = {std_m * std_m, std_m * std_m, std_m * std_m};
	const bool refresh_optical_anchor =
	    association_position_only_refreshes_optical_anchor(hyp, used_pnp_position);
	constellation_tracked_device_connection_notify_position(device->connection, sample->timestamp, dev_state->estimator_prior,
	                                                        &position, &position_variance,
	                                                        refresh_optical_anchor);

	if (g2_telem_enabled()) {
		const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;
		association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp, hyp,
		                           true, 0, NULL);
		g2_telem_event(telem_device_id(xdev), sample->timestamp, G2_TELEM_EV_ASSOC_POSITION_ONLY_SELECTED,
		               std_m);
	}

	return true;
}

static const struct association_pose_hypothesis *
association_find_sibling_twin(const struct association_device_work *work,
                              const struct association_pose_hypothesis *chosen,
                              bool want_pure_yaw,
                              struct xrt_quat *out_twin_world)
{
	if (work == NULL || chosen == NULL || (chosen->flags & ASSOC_HYP_HAS_TWIN) == 0) {
		return NULL;
	}

	const uint16_t chosen_is_twin = chosen->flags & ASSOC_HYP_IS_TWIN;
	const struct xrt_vec3 world_up = {0.f, 1.f, 0.f};
	for (int h = 0; h < work->count; h++) {
		const struct association_pose_hypothesis *hyp = &work->hyps[h];
		if (hyp == chosen || hyp->primary_view_id != chosen->primary_view_id || hyp->source != chosen->source ||
		    (hyp->flags & ASSOC_HYP_HAS_TWIN) == 0 || (hyp->flags & ASSOC_HYP_IS_TWIN) == chosen_is_twin) {
			continue;
		}

		double tilt_rad = 0.0;
		double yaw_rad = 0.0;
		pose_metrics_prior_orient_split(&hyp->pose_world.orientation, &chosen->pose_world.orientation,
		                                &world_up, &tilt_rad, &yaw_rad);
		const bool is_pure_yaw = tilt_rad < (double)ASSOC_TEMPORAL_TWIN_MIN_TILT_SEP_RAD;
		if (is_pure_yaw != want_pure_yaw) {
			continue;
		}
		if (out_twin_world != NULL) {
			*out_twin_world = hyp->pose_world.orientation;
		}
		return hyp;
	}
	return NULL;
}

static bool
association_pure_yaw_ambiguous(const struct association_device_work *work,
                               const struct association_pose_hypothesis *chosen,
                               const struct association_pose_hypothesis **out_twin)
{
	if (chosen == NULL || (chosen->flags & ASSOC_HYP_JOINT) != 0 || (chosen->flags & ASSOC_HYP_HAS_TWIN) == 0 ||
	    !association_lock_eligible(chosen)) {
		return false;
	}

	const struct association_pose_hypothesis *twin =
	    association_find_sibling_twin(work, chosen, /*want_pure_yaw*/ true, NULL);
	if (twin == NULL || !association_lock_eligible(twin)) {
		return false;
	}

	if (fabsf(chosen->cost.total_nll - twin->cost.total_nll) > ASSOC_YAW_BELIEF_TIE_NLL ||
	    fabsf(chosen->cost.reprojection_nll - twin->cost.reprojection_nll) > ASSOC_YAW_BELIEF_TIE_NLL ||
	    fabsf(chosen->cost.missed_led_nll - twin->cost.missed_led_nll) > ASSOC_YAW_BELIEF_TIE_NLL ||
	    fabsf(chosen->cost.temporal_nll - twin->cost.temporal_nll) > ASSOC_YAW_BELIEF_TIE_NLL ||
	    chosen->matched_count != twin->matched_count) {
		return false;
	}

	if (out_twin != NULL) {
		*out_twin = twin;
	}
	return true;
}

static void
association_seed_yaw_belief(struct t_constellation_tracker *ct,
                            const struct tracking_sample_device_state *dev_state,
                            const struct constellation_tracking_sample *sample,
                            const struct association_pose_hypothesis *chosen,
                            const struct association_pose_hypothesis *twin)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	yaw_belief_seed(&device->yaw_belief, sample->timestamp, &chosen->pose_world.orientation,
	                &twin->pose_world.orientation, &dev_state->P_world_obj_prior.orientation,
	                dev_state->prior_tilt_trusted);
}

static const struct association_pose_hypothesis *
association_best_position_only_local(const struct association_device_work *work,
                                     const struct association_pose_hypothesis *const chosen[],
                                     int chosen_count)
{
	const struct association_pose_hypothesis *best = NULL;
	float best_cost = INFINITY;
	for (int h = 0; work != NULL && h < work->count; h++) {
		const struct association_pose_hypothesis *hyp = &work->hyps[h];
		if (!association_position_only_eligible(hyp) ||
		    !association_is_compatible_with_chosen(hyp, chosen, chosen_count)) {
			continue;
		}
		const float cost = association_option_cost(hyp, ASSOC_OBS_POSITION_ONLY);
		if (cost < best_cost) {
			best = hyp;
			best_cost = cost;
		}
	}
	return best;
}

static enum association_observation_kind
association_update_yaw_belief(struct t_constellation_tracker *ct,
                              const struct association_device_work *work,
                              struct tracking_sample_device_state *dev_state,
                              const struct constellation_tracking_sample *sample,
                              const struct association_pose_hypothesis *const chosen_others[],
                              int chosen_count,
                              const struct association_pose_hypothesis **out_commit_hyp)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct constellation_yaw_belief belief = device->yaw_belief;
	if (!belief.active) {
		return ASSOC_OBS_ABSENT;
	}

	struct xrt_quat prop[2];
	struct yaw_belief_mode_evidence ev[2] = {0};
	const struct association_pose_hypothesis *cand[2] = {NULL, NULL};
	for (int mode = 0; mode < 2; mode++) {
		prop[mode] = belief.mode_q[mode];
		if (belief.seed_prior_trusted) {
			prop[mode] = association_propagate_temporal_ref(&belief.mode_q[mode], &belief.seed_prior,
			                                                &dev_state->P_world_obj_prior.orientation);
		}
	}

	float best_match_nll[2] = {INFINITY, INFINITY};
	for (int h = 0; work != NULL && h < work->count; h++) {
		const struct association_pose_hypothesis *hyp = &work->hyps[h];
		if (hyp->primary_view_id < 0 || !association_lock_eligible(hyp) ||
		    !association_is_compatible_with_chosen(hyp, chosen_others, chosen_count)) {
			continue;
		}
		for (int mode = 0; mode < 2; mode++) {
			const float match =
			    (float)association_capped_yaw_continuity_nll(&hyp->pose_world.orientation, &prop[mode]) +
			    hyp->cost.total_nll;
			if (match < best_match_nll[mode]) {
				best_match_nll[mode] = match;
				cand[mode] = hyp;
				ev[mode].have_cand = true;
				ev[mode].cand_q = hyp->pose_world.orientation;
				ev[mode].cand_total_nll = hyp->cost.total_nll;
			}
		}
	}

	const struct yaw_belief_params params = {
	    .commit_w = ASSOC_YAW_BELIEF_COMMIT_W,
	    .no_cand_nll = ASSOC_TEMPORAL_MAX_NLL,
	    .ttl_ns = (uint64_t)ASSOC_TEMPORAL_YAW_TTL_NS,
	    .max_frames = ASSOC_YAW_BELIEF_MAX_FRAMES,
	    .yaw_sigma_rad = ASSOC_TEMPORAL_YAW_SIGMA_RAD,
	    .tilt_sigma_rad = ASSOC_TEMPORAL_TILT_SIGMA_RAD,
	    .huber_knee = FLIP_COST_HUBER_KNEE_SIGMA,
	    .weight = FLIP_COST_WEIGHT,
	    .cap = ASSOC_TEMPORAL_MAX_NLL,
	};

	int commit_mode = 0;
	enum yaw_belief_action action = YAW_BELIEF_EXPIRE;
	if (device->yaw_belief.active && device->yaw_belief.seed_ts == belief.seed_ts) {
		action = yaw_belief_step(&device->yaw_belief, sample->timestamp, prop, ev, &params, &commit_mode);
	}

	if (action == YAW_BELIEF_COMMIT_MODE) {
		*out_commit_hyp = cand[commit_mode];
		return *out_commit_hyp != NULL ? ASSOC_OBS_POSE_LOCK : ASSOC_OBS_ABSENT;
	}
	if (action == YAW_BELIEF_DEFER) {
		const struct association_pose_hypothesis *pos =
		    association_best_position_only_local(work, chosen_others, chosen_count);
		if (pos != NULL) {
			*out_commit_hyp = pos;
			return ASSOC_OBS_POSITION_ONLY;
		}
		const struct association_pose_hypothesis *fold =
		    association_best_lockable_local(work, chosen_others, chosen_count);
		if (fold != NULL && association_led_fold_eligible(fold)) {
			*out_commit_hyp = fold;
			return ASSOC_OBS_LED_FOLD;
		}
	}

	return ASSOC_OBS_ABSENT;
}

static void
constellation_associate_covariance_frame(struct t_constellation_tracker *ct,
                                         struct constellation_tracking_sample *sample)
{
	struct association_device_work work[CONSTELLATION_MAX_DEVICES] = {0};
	for (int i = 0; i < sample->n_devices; i++) {
		association_build_device_work(ct, &work[i], &sample->devices[i], sample);
	}
	ct->last_assoc_work_units = association_run_cold_waves(ct, work, sample);
	for (int i = 0; i < sample->n_devices; i++) {
		association_apply_orientation_consensus(&work[i]);
	}
	association_apply_joint_contention(ct, work, sample);

	const struct association_pose_hypothesis *current[CONSTELLATION_MAX_DEVICES] = {0};
	enum association_observation_kind current_kind[CONSTELLATION_MAX_DEVICES] = {0};
	struct association_joint_choice best = {0};
	association_select_joint_recursive(work, sample, sample->n_devices, 0, current, current_kind, 0.0f, -1, NULL,
	                                   &best);
	if (!best.valid) {
		for (int i = 0; i < sample->n_devices; i++) {
			association_fold_prior_leds(ct, &sample->devices[i], sample);
		}
		return;
	}

	/* Identity-ambiguity commitment discipline. A visual observation may only feed the filter when
	 * the joint evidence actually pins the cluster's DEVICE IDENTITY: re-solve the joint selection
	 * with this device banned from its chosen cluster, and if a PARTNER takes THAT SAME CLUSTER in
	 * an alternative assignment costing < ASSOC_IDENTITY_DEFER_MARGIN_NLL extra, the identity is a
	 * near coin-flip (at 4-7 bloomed LEDs both chiral models fit and the tilt clamp can erase the
	 * gravity objection). Committing would collapse the ESKF covariance onto a possibly-wrong ring
	 * and poison every later prior objection into DEFENDING the swap, so the observation defers
	 * until accumulating LEDs separate the permutations. Whether to defer is priced purely by the
	 * evidence margin; geometry only identifies WHICH pair of claims are rivals.
	 * "Same cluster" means the two claims place their device at the same POINT, within
	 * the identity discrimination margin. A shared blob does not answer it in either direction:
	 * two adjacent controllers contend for one boundary blob while each sits on its own prior
	 * (xv1 t=52.122 s, 1 shared blob of 16, poses 10.2 cm apart, both priors within 3.7 cm of
	 * their own device -- contention, not identity), and two poses on ONE ring split its blobs
	 * between them under joint exclusivity (xv1 t=60.996 s, 1 shared blob of 5, poses 2.1 cm
	 * apart, the claim 18.1 cm from its own prior and 0.7 cm from the partner's -- a steal).
	 * Measured separations: 2.1 / 2.2 / 3.0 cm on the real bifurcations against 10.2 cm on the
	 * contention frame. */
	bool identity_deferred[CONSTELLATION_MAX_DEVICES] = {false};
	for (int slot = 0; slot < sample->n_devices; slot++) {
		if (sample->n_devices < 2 || best.chosen[slot] == NULL ||
		    !association_observation_kind_visual(best.kind[slot])) {
			continue;
		}
		struct association_joint_choice alt = {0};
		association_select_joint_recursive(work, sample, sample->n_devices, 0, current, current_kind, 0.0f,
		                                   slot, best.chosen[slot], &alt);
		if (!alt.valid || alt.total_cost - best.total_cost >= ASSOC_IDENTITY_DEFER_MARGIN_NLL) {
			continue;
		}
		for (int other = 0; other < sample->n_devices; other++) {
			if (other == slot || !association_observation_kind_visual(alt.kind[other]) ||
			    !association_hypotheses_same_cluster(alt.chosen[other], best.chosen[slot],
			                                         ASSOC_IDENTITY_STEAL_MARGIN_M)) {
				continue;
			}
			identity_deferred[slot] = true;
			if (g2_telem_enabled()) {
				const struct constellation_tracker_device *cdev =
				    ct->devices + sample->devices[slot].dev_index;
				const struct xrt_device *xdev =
				    cdev->connection != NULL ? cdev->connection->xdev : NULL;
				g2_telem_event(telem_device_id(xdev), sample->timestamp,
				               G2_TELEM_EV_ASSOC_IDENTITY_DEFER,
				               alt.total_cost - best.total_cost);
			}
			break;
		}
	}

	/* The joint selection guarantees mutual blob-exclusivity of best.chosen; a yaw-belief override
	 * replaces a slot's choice afterwards, so it must honor the same constraint against what the other
	 * slots actually take (committed tracks overrides as the loop resolves each slot). */
	const struct association_pose_hypothesis *committed[CONSTELLATION_MAX_DEVICES] = {0};
	for (int slot = 0; slot < sample->n_devices; slot++) {
		committed[slot] = best.chosen[slot];
	}

	for (int slot = 0; slot < sample->n_devices; slot++) {
		struct tracking_sample_device_state *dev_state = sample->devices + slot;
		struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
		const struct association_pose_hypothesis *choice = best.chosen[slot];
		enum association_observation_kind choice_kind = best.kind[slot];
		const struct association_pose_hypothesis *best_lockable =
		    association_best_lockable_local(&work[slot], NULL, 0);

		if (identity_deferred[slot]) {
			/* No visual commit AND no prior-LED folds: a contested slot's prior labels may
			 * themselves be swap-poisoned, and one frame of coast is the honest price of an
			 * unresolved identity. */
			committed[slot] = NULL;
			continue;
		}

		if (device->yaw_belief.active) {
			committed[slot] = NULL;
			const struct association_pose_hypothesis *belief_choice = NULL;
			const enum association_observation_kind belief_kind = association_update_yaw_belief(
			    ct, &work[slot], dev_state, sample, committed, sample->n_devices, &belief_choice);
			if (device->yaw_belief.active || belief_kind != ASSOC_OBS_ABSENT) {
				choice = belief_choice;
				choice_kind = belief_kind;
			}
		} else if (choice_kind == ASSOC_OBS_POSE_LOCK) {
			const struct association_pose_hypothesis *twin = NULL;
			if (association_pure_yaw_ambiguous(&work[slot], choice, &twin) &&
			    association_position_only_eligible(choice)) {
				association_seed_yaw_belief(ct, dev_state, sample, choice, twin);
				choice_kind = ASSOC_OBS_POSITION_ONLY;
			}
		}

		committed[slot] = choice;

		if (choice == NULL && best_lockable != NULL) {
			association_emit_lockable_not_chosen(ct, dev_state, sample, best_lockable, best.chosen, slot);
		}
		if (choice_kind == ASSOC_OBS_POSE_LOCK) {
			if (association_commit_joint_pose(ct, dev_state, sample, choice)) {
				continue;
			}
		}

		if (choice_kind == ASSOC_OBS_POSITION_ONLY) {
			if (association_commit_position_only(ct, dev_state, sample, choice)) {
				work[slot].folded_partial = true;
				continue;
			}
		}

		if (choice_kind == ASSOC_OBS_LED_FOLD) {
			association_fold_ambiguous_hypothesis(ct, dev_state, sample, choice);
			work[slot].folded_partial = true;
			continue;
		}

		if (association_fold_prior_leds(ct, dev_state, sample)) {
			work[slot].folded_partial = true;
		}
	}
}

/* Project a device's LED model at the given world pose into this camera's distorted image and
 * append the in-bounds pixels to the static map's exemption list (same projection convention as
 * pose_metrics). All LEDs are used, facing or not — the exemption protects a REGION around the
 * device, not a visibility prediction. */
static void
append_exempt_led_px(const struct tracking_sample_frame *view,
                     const struct constellation_tracker_camera_state *cam,
                     const struct t_constellation_led_model *led_model,
                     const struct xrt_pose *P_world_obj,
                     struct xrt_vec2 *exempt_px,
                     int *n_exempt,
                     int max_exempt)
{
	struct xrt_pose P_cam_obj;
	math_pose_transform(&view->P_cam_world, P_world_obj, &P_cam_obj);
	for (int li = 0; li < led_model->num_leds && *n_exempt < max_exempt; li++) {
		struct xrt_vec3 cam_point;
		math_pose_transform_point(&P_cam_obj, &led_model->leds[li].pos, &cam_point);
		if (cam_point.z <= 0.0f) {
			continue;
		}
		float u, v;
		if (!t_camera_models_project(&cam->camera_model.calib, cam_point.x, cam_point.y, cam_point.z, &u,
		                             &v)) {
			continue;
		}
		if (u < 0.0f || v < 0.0f || u >= (float)cam->camera_model.width ||
		    v >= (float)cam->camera_model.height) {
			continue;
		}
		exempt_px[(*n_exempt)++] = (struct xrt_vec2){u, v};
	}
}

/* Project one world-space device pose into a camera and union its bounding rect (clamped to the
 * image by pose_metrics_get_device_bounds) into *rect (replacing it when !have_rect). Returns
 * true iff the pose contributed a rect with area (some bounding point in front of the camera
 * and projectable). */
static bool
mask_union_device_rect(const struct tracking_sample_frame *view,
                       struct constellation_tracker_camera_state *cam,
                       struct t_constellation_led_model *led_model,
                       const struct xrt_pose *P_world_obj,
                       bool have_rect,
                       struct pose_rect *rect,
                       float *out_depth_m)
{
	struct xrt_pose P_cam_obj;
	math_pose_transform(&view->P_cam_world, P_world_obj, &P_cam_obj);

	struct pose_rect bounds;
	pose_metrics_get_device_bounds(&P_cam_obj, led_model, &cam->camera_model, &bounds, NULL, NULL);
	if (!pose_rect_has_area(&bounds)) {
		return false;
	}
	if (out_depth_m != NULL) {
		*out_depth_m = P_cam_obj.position.z;
	}
	if (!have_rect) {
		*rect = bounds;
		return true;
	}
	rect->left = fmin(rect->left, bounds.left);
	rect->top = fmin(rect->top, bounds.top);
	rect->right = fmax(rect->right, bounds.right);
	rect->bottom = fmax(rect->bottom, bounds.bottom);
	return true;
}

/* Reuse the matcher's optical geometry horizon. Updating a rectangle every camera frame does
 * not make an old optical observation current, and a future timestamp is not valid evidence. */
static bool
controller_mask_observation_fresh(uint64_t frame_ts, bool have_last_seen_pose, uint64_t last_seen_pose_ts)
{
	return have_last_seen_pose && frame_ts >= last_seen_pose_ts &&
	       frame_ts - last_seen_pose_ts <= (uint64_t)(ROI_MAX_OPTICAL_AGE_MS * U_TIME_1MS_IN_NS);
}

/* B5 mask repair: push the SLAM controller masks for this frame, one rect per connected device
 * per camera, from the live prediction + last-seen pose (see the MASK_* constants' comment for
 * the measured rationale). Runs at frame cadence under the tracked-device lock, so mask staleness
 * is bounded by one controller-frame period instead of the time since the last optical accept,
 * the rect follows the controller through coast, and every rect is re-projected through the LIVE
 * head pose (a world-anchored rect, never a pixel-frozen one). Devices with neither a bounded
 * prediction nor a last-seen pose are pushed disabled — no rect ever outlives its knowledge. */
static void
push_controller_masks(struct t_constellation_tracker *ct, struct constellation_tracking_sample *sample)
{
	if (ct->controller_masks_sink == NULL) {
		return;
	}

	struct xrt_device_masks_sample *masks = &ct->controller_masks_sample;
	*masks = (struct xrt_device_masks_sample){0};

	uint8_t flags[CONSTELLATION_MAX_CAMERAS][CONSTELLATION_MAX_DEVICES] = {0};
	float sigma_px[CONSTELLATION_MAX_CAMERAS][CONSTELLATION_MAX_DEVICES];

	for (int i = 0; i < sample->n_views; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		struct tracking_sample_frame *view = sample->views + i;
		struct xrt_device_masks_sample_camera *sample_camera = &masks->views[cam->slam_tracking_index];

		for (int d = 0; d < sample->n_devices; d++) {
			struct tracking_sample_device_state *dev_state = sample->devices + d;
			struct xrt_device_masks_sample_device *device_mask =
			    &sample_camera->devices[dev_state->dev_index];

			const struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
			const bool optical_fresh = controller_mask_observation_fresh(
			    sample->timestamp, dev_state->have_last_seen_pose, device->last_seen_pose_ts);
			bool have_rect = false;
			struct pose_rect rect = {0, 0, 0, 0};
			sigma_px[i][d] = -1.0f;

			/* Prediction rect, inflated by the projected position uncertainty. Trusted
			 * exactly as far as the fusion can bound it: a coasting controller gets a
			 * honestly-larger rect, a lost/divergent one inflates past the area cap below
			 * and self-disables. */
			if (dev_state->prior_pos_std_m >= 0.0f &&
			    (dev_state->prior_position_tracked || optical_fresh)) {
				float depth_m = MASK_SIGMA_MIN_DEPTH_M;
				if (mask_union_device_rect(view, cam, dev_state->led_model,
				                           &dev_state->P_world_obj_prior, have_rect, &rect,
				                           &depth_m)) {
					have_rect = true;
					const float focal = fmaxf(cam->camera_model.calib.fx, cam->camera_model.calib.fy);
					const float sigma = focal * (float)PRIOR_GATE_SIGMA * dev_state->prior_pos_std_m /
					                    fmaxf(depth_m, MASK_SIGMA_MIN_DEPTH_M);
					rect.left -= sigma;
					rect.top -= sigma;
					rect.right += sigma;
					rect.bottom += sigma;
					sigma_px[i][d] = sigma;
					flags[i][d] |= G2_TELEM_MASK_FROM_PREDICTION;
				}
			}

			/* Fresh observed geometry remains useful during a short coast or confirmed idle,
			 * even when the estimator reports position untracked. An expired observation must
			 * not keep excluding scene features indefinitely. */
			if (optical_fresh &&
			    mask_union_device_rect(view, cam, dev_state->led_model, &dev_state->last_seen_pose,
			                           have_rect, &rect, NULL)) {
				have_rect = true;
				flags[i][d] |= G2_TELEM_MASK_FROM_LAST_SEEN;
			}

			if (!have_rect) {
				continue;
			}

			rect.left = fmax(rect.left - MASK_HALO_MARGIN_PX, 0.0);
			rect.top = fmax(rect.top - MASK_HALO_MARGIN_PX, 0.0);
			rect.right = fmin(rect.right + MASK_HALO_MARGIN_PX, (double)cam->camera_model.width - 1.0);
			rect.bottom = fmin(rect.bottom + MASK_HALO_MARGIN_PX, (double)cam->camera_model.height - 1.0);
			if (!pose_rect_has_area(&rect) || rect.right < rect.left || rect.bottom < rect.top) {
				continue;
			}

			device_mask->enabled = true;
			device_mask->rect = (struct xrt_rect_f32){
			    .x = (float)rect.left,
			    .y = (float)rect.top,
			    .w = (float)(rect.right - rect.left),
			    .h = (float)(rect.bottom - rect.top),
			};
			flags[i][d] |= G2_TELEM_MASK_ENABLED;
		}

		/* Per-device area cap: a rect that must be giant to cover its uncertainty is
		 * worthless as a mask — disable it; the unmasked light of an unpredictable
		 * controller is almost always outside the SLAM cams' view too. Deliberately NOT a
		 * total-sum cap: rationing two honest close-range rects against each other unmasks
		 * an in-view controller (measured leak regression, see the MASK_* constants). */
		const float cap_px2 =
		    MASK_AREA_CAP_FRACTION * (float)cam->camera_model.width * (float)cam->camera_model.height;
		for (int d = 0; d < sample->n_devices; d++) {
			struct xrt_device_masks_sample_device *device_mask =
			    &sample_camera->devices[sample->devices[d].dev_index];
			if (!device_mask->enabled || device_mask->rect.w * device_mask->rect.h <= cap_px2) {
				continue;
			}
			device_mask->enabled = false;
			flags[i][d] &= (uint8_t)~G2_TELEM_MASK_ENABLED;
			flags[i][d] |= G2_TELEM_MASK_AREA_CAPPED;
		}

		if (g2_telem_enabled()) {
			for (int d = 0; d < sample->n_devices; d++) {
				struct tracking_sample_device_state *dev_state = sample->devices + d;
				struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
				const struct xrt_device_masks_sample_device *device_mask =
				    &sample_camera->devices[dev_state->dev_index];
				const float mrect[4] = {
				    device_mask->rect.x,
				    device_mask->rect.y,
				    device_mask->rect.x + device_mask->rect.w,
				    device_mask->rect.y + device_mask->rect.h,
				};
				const float optical_age_ms =
				    device->have_last_seen_pose
				        ? (float)((int64_t)(sample->timestamp - device->last_seen_pose_ts) / 1e6)
				        : -1.0f;
				g2_telem_mask((uint8_t)cam->slam_tracking_index, sample->timestamp,
				              telem_device_id(device->connection->xdev), flags[i][d], mrect,
				              sigma_px[i][d], optical_age_ms);
			}
		}
	}

	xrt_sink_push_device_masks(ct->controller_masks_sink, masks);
}

/* The device estimator uses OpenXR world coordinates; frontend optical caches use the
 * sandwich-flipped OpenCV world. Rebase both in the same transaction before this frame
 * takes its priors/camera geometry. Preserve observation timestamps: a rebase is not a
 * new optical observation. Caller holds no tracked-device or connection lock. */
static void
constellation_reanchor_devices(struct t_constellation_tracker *ct,
                                timepoint_ns frame_mono_ns,
                                const struct xrt_pose *delta,
                                const struct xrt_vec3 *new_raw_pivot,
                                timepoint_ns publication_ns,
                                double gyro_dps,
                                double speed_mps)
{
	struct xrt_pose delta_cv;
	pose_flip_YZ(delta, &delta_cv);
	os_mutex_lock(&ct->tracked_device_lock);
	for (int i = 0; i < ct->num_devices; i++) {
		struct constellation_tracker_device *device = ct->devices + i;
		if (device->have_last_seen_pose) {
			struct xrt_pose rebased;
			math_pose_transform(&delta_cv, &device->last_seen_pose, &rebased);
			device->last_seen_pose = rebased;
		}
		if (device->connection != NULL) {
			constellation_tracked_device_connection_notify_world_reanchor(
			    device->connection, frame_mono_ns, delta, new_raw_pivot, publication_ns, gyro_dps, speed_mps);
		}
	}
	os_mutex_unlock(&ct->tracked_device_lock);
}

// Fast frame processing: blob extraction and match to existing predictions
static void
constellation_tracker_process_frame_fast(struct xrt_frame_sink *sink, struct xrt_frame *xf)
{
	struct t_constellation_tracker *ct = container_of(sink, struct t_constellation_tracker, fast_process_sink);
	struct xrt_space_relation xsr_base_pose = {0};

	/* Allocate a tracking sample for everything we're about to process */
	struct constellation_tracking_sample *sample = constellation_tracking_sample_new();
	if (sample == NULL) {
		CT_ERROR(ct, "Failed to allocate tracking sample; dropping frame %" PRIu64, xf->source_sequence);
		return;
	}
	uint64_t fast_analysis_start_ts = os_monotonic_get_ns();

	CT_DEBUG(ct, "Starting analysis of frame %" PRIu64 " TS %" PRIu64, xf->source_sequence, xf->timestamp);

	/* Get the HMD's pose so we can calculate the camera view poses */
	xrt_device_get_tracked_pose(ct->hmd_xdev, XRT_INPUT_GENERIC_TRACKER_POSE, xf->timestamp, &xsr_base_pose);

	/* Record the live SLAM head pose for this frame (world, OpenXR) so the offline replay harness can
	 * reproduce the true camera->world transform instead of an IMU-only reconstruction. */
	if (g2_telem_enabled()) {
		float head7[7];
		telem_pack_pose(&xsr_base_pose.pose, head7);
		g2_telem_head_pose((uint64_t)xf->timestamp, head7);
	}

	/* World re-anchor detection (B2 complement): a SLAM relocalization/reset steps THIS sampled
	 * head pose, and with it every P_world_cam below — the controllers' world moves in the same
	 * camera frame. Detect the step's excess beyond the IMU envelope (same math as the
	 * presentation guard) and re-anchor each device's fusion world-state by the implied rigid
	 * delta BEFORE this frame's observations are matched/folded, so the ESKF prior lands in the
	 * new world the observations live in instead of disagreeing by the full jump. */
	if ((xsr_base_pose.relation_flags &
	     (XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_POSITION_VALID_BIT |
	      XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT)) ==
	    (XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_POSITION_VALID_BIT |
	     XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT)) {
		const struct xrt_vec3 *av = &xsr_base_pose.angular_velocity;
		double gyro_dps = sqrt((double)av->x * av->x + (double)av->y * av->y + (double)av->z * av->z) *
		                  (180.0 / M_PI);
		if (ct->reanchor_have_prev && xf->timestamp > ct->reanchor_prev_ts) {
			double dt_s = (double)(xf->timestamp - ct->reanchor_prev_ts) * 1e-9;
			double gyro_int_deg = 0.5 * (ct->reanchor_prev_gyro_dps + gyro_dps) * dt_s;
			struct u_world_reanchor_step st;
			if (u_world_reanchor_compute_excess(&u_world_reanchor_default_params,
			                                    &ct->reanchor_prev_pose, &xsr_base_pose.pose, dt_s,
			                                    gyro_int_deg, &st)) {
				struct xrt_pose delta;
				u_world_reanchor_step_to_world_delta(&st, &ct->reanchor_prev_pose.position,
				                                     &delta);
				CT_DEBUG(ct,
				         "World re-anchor at frame TS %" PRIi64
				         ": step %.2f deg / %.1f mm, excess %.2f deg / %.1f mm -> re-anchoring "
				         "device fusion world state",
				         xf->timestamp, st.dang_deg, st.dnorm_m * 1e3, st.exc_ang_deg,
				         st.exc_pos_m * 1e3);
				// One publication clock and pivot for this exact raw transition on both hands.
				const timepoint_ns publication_ns = os_monotonic_get_ns();
				struct xrt_pose new_pivot_pose = { .orientation = {0, 0, 0, 1},
				                                  .position = ct->reanchor_prev_pose.position };
				struct xrt_pose transformed_pivot;
				math_pose_transform(&delta, &new_pivot_pose, &transformed_pivot);
				const struct xrt_vec3 *lv = &xsr_base_pose.linear_velocity;
				const double speed_mps = sqrt((double)lv->x*lv->x + (double)lv->y*lv->y + (double)lv->z*lv->z);
				constellation_reanchor_devices(ct, (timepoint_ns)xf->timestamp, &delta,
				                               &transformed_pivot.position, publication_ns, gyro_dps, speed_mps);
			}
		}
		ct->reanchor_prev_pose = xsr_base_pose.pose;
		ct->reanchor_prev_ts = xf->timestamp;
		ct->reanchor_prev_gyro_dps = gyro_dps;
		ct->reanchor_have_prev = true;
	}

	/* Split out camera views and collect blobs across all cameras */
	assert(ct->cam_count <= XRT_TRACKING_MAX_SLAM_CAMS);
	sample->n_views = ct->cam_count;
	sample->timestamp = xf->timestamp;
	/* One prior per device BEFORE any ROI, blob association, or view fold. The frame consumer is
	 * serial and waits for cold workers before finishing; camera geometry and these tokens stay paired. */
	os_mutex_lock(&ct->tracked_device_lock);
	assert(ct->num_devices <= CONSTELLATION_MAX_DEVICES);
	for (int d = 0; d < ct->num_devices; ++d) {
		sample->estimator_prior_supported[d] = constellation_tracked_device_connection_get_estimator_prior(
		    ct->devices[d].connection, xf->timestamp, &sample->estimator_priors[d]);
	}
	os_mutex_unlock(&ct->tracked_device_lock);

	for (int i = 0; i < ct->cam_count; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		struct tracking_sample_frame *view = sample->views + i;

		// Flip the input pose to CV coords, so we can do all our operations
		// in OpenCV coords
		struct xrt_pose P_cvworld_hmdimu;
		pose_flip_YZ(&xsr_base_pose.pose, &P_cvworld_hmdimu);

		math_pose_transform(&P_cvworld_hmdimu, &cam->P_imu_cam, &view->P_world_cam);

		CT_DEBUG(ct,
		         "Prepare transforms for cam %d "
		         " HMD pose %f,%f,%f,%f pos %f,%f,%f "
		         " P_imu_cam %f,%f,%f,%f pos %f,%f,%f "
		         " P_world_cam %f,%f,%f,%f pos %f,%f,%f ",
		         i, xsr_base_pose.pose.orientation.x, xsr_base_pose.pose.orientation.y,
		         xsr_base_pose.pose.orientation.z, xsr_base_pose.pose.orientation.w,
		         xsr_base_pose.pose.position.x, xsr_base_pose.pose.position.y, xsr_base_pose.pose.position.z,

		         cam->P_imu_cam.orientation.x, cam->P_imu_cam.orientation.y, cam->P_imu_cam.orientation.z,
		         cam->P_imu_cam.orientation.w, cam->P_imu_cam.position.x, cam->P_imu_cam.position.y,
		         cam->P_imu_cam.position.z,

		         view->P_world_cam.orientation.x, view->P_world_cam.orientation.y,
		         view->P_world_cam.orientation.z, view->P_world_cam.orientation.w, view->P_world_cam.position.x,
		         view->P_world_cam.position.y, view->P_world_cam.position.z);

		// Calculate inverse from cam back to world coords
		math_pose_invert(&view->P_world_cam, &view->P_cam_world);

		const struct xrt_vec3 gravity_vector = {0.0, 1.0, 0.0};
		math_quat_rotate_vec3(&view->P_cam_world.orientation, &gravity_vector, &view->cam_gravity_vector);

		u_frame_create_roi(xf, cam->roi, &view->vframe);
		view->bw = cam->bw;

		/* The actual exposure for this frame is encoded in the full frame's pixel header
		 * (wmr_camera.c: data[6..7]); read it from xf, not the ROI sub-frame which excludes
		 * the header row. Same value for all cameras on a controller frame. */
		uint16_t frame_exposure =
		    (xf->data != NULL && xf->size > 7) ? (uint16_t)((xf->data[6] << 8) | xf->data[7]) : 0;

		/* Predictive-ROI: project every LED of every tracked device through its ESKF state to this
		 * camera's image, build a bounding box, restrict blob search to it. Falls back to full-frame
		 * when any connected device lacks enough in-frame predictions (so one tracked controller cannot
		 * crop out a controller that needs full-frame acquisition/re-acquisition). The scan pad is
		 * per-LED-adaptive (max of the base pad floor and k * sqrt(max-diag(S))). */
		bool used_roi = false;
		{
			/* Build the cam_calib + the OpenXR-camera-frame pose this camera uses for projection, the
			 * same way emit_view_led_observations does so the predicted pixels live in the same coords
			 * as the LED gate uses elsewhere in the tracker. */
			struct xrt_pose P_xrworld_cam_pred;
			math_pose_transform(&view->P_cam_world, &P_YZ_FLIP, &P_xrworld_cam_pred);
			const struct t_constellation_cam_calib cam_calib_pred = {
			    cam->camera_model.calib.fx, cam->camera_model.calib.fy,
			    cam->camera_model.calib.cx, cam->camera_model.calib.cy};
				float xmin = 1e9f, ymin = 1e9f, xmax = -1e9f, ymax = -1e9f;
				int n_in_frame = 0;
				int n_roi_ready_devices = 0;
				int n_connected_devices = 0;
				for (int d = 0; d < ct->num_devices; d++) {
					struct constellation_tracker_device *device = &ct->devices[d];
					const struct t_constellation_led_model *lm = &device->led_model;
						if (device->connection == NULL || lm->leds == NULL || lm->num_leds == 0) {
							continue; /* not connected / model not loaded yet */
						}
						n_connected_devices++;
						double optical_age_ms = 0.0;
						if (sample->estimator_prior_supported[d]) {
							optical_age_ms = sample->estimator_priors[d].optical_age_ms;
							if (!sample->estimator_priors[d].valid) { continue; }
						} else {
							constellation_tracked_device_connection_get_last_optical_age_ms(
							    device->connection, xf->timestamp, &optical_age_ms);
						}
						if (optical_age_ms > ROI_MAX_OPTICAL_AGE_MS) {
							continue;
						}
						int device_in_frame = 0;
					for (int li = 0; li < lm->num_leds; li++) {
						struct xrt_vec3 led_flip = {lm->leds[li].pos.x, -lm->leds[li].pos.y,
						                            -lm->leds[li].pos.z};
					struct xrt_vec3 led_obj;
					math_pose_transform_point(&lm->P_device_model, &led_flip, &led_obj);
					float zhat[2], S[4];
						if (!constellation_tracked_device_connection_predict_led_gate(
						        device->connection, xf->timestamp,
						        sample->estimator_prior_supported[d] ? &sample->estimator_priors[d] : NULL, &P_xrworld_cam_pred,
						        &cam_calib_pred, &led_obj, zhat, S)) {
							continue; /* device untracked: no usable prior */
						}
						if (!isfinite(zhat[0]) || !isfinite(zhat[1]) || !isfinite(S[0]) || !isfinite(S[3])) {
							continue; /* non-finite projection or covariance: skip */
						}
						/* zhat is in FULL-camera pixel coords (cam_calib's principal point);
						 * translate to view->vframe coords by subtracting the ROI offset. */
					const float vx = zhat[0] - (float)cam->roi.offset.w;
					const float vy = zhat[1] - (float)cam->roi.offset.h;
					if (vx >= 0.f && vx < (float)view->vframe->width && vy >= 0.f &&
					    vy < (float)view->vframe->height) {
						/* Per-LED adaptive pad: sqrt of the max-diagonal of S is an upper bound on
						 * the 1-sigma pixel-space prediction noise (max eigenvalue ≤ trace ≤ 2*max
						 * diag). k·σ gives the half-width of the LED's plausibility patch; floored
						 * at the base pad so a near-zero S can't shrink it below the centroid noise. */
						const float sx = S[0] > 0.0f ? sqrtf(S[0]) : 0.0f;
						const float sy = S[3] > 0.0f ? sqrtf(S[3]) : 0.0f;
						const float pix_sigma = sx > sy ? sx : sy;
						float per_led_pad = ROI_SIGMA_K * pix_sigma;
						if (per_led_pad < (float)ROI_BASE_PAD_PX) {
							per_led_pad = (float)ROI_BASE_PAD_PX;
						}
						const float lx = vx - per_led_pad;
						const float ly = vy - per_led_pad;
						const float hx = vx + per_led_pad;
						const float hy = vy + per_led_pad;
						if (lx < xmin) xmin = lx;
						if (ly < ymin) ymin = ly;
							if (hx > xmax) xmax = hx;
							if (hy > ymax) ymax = hy;
							n_in_frame++;
							device_in_frame++;
						}
					}
					if (device_in_frame >= ROI_FALLBACK_MIN_PREDICTIONS) {
						n_roi_ready_devices++;
					}
				}
				if (n_connected_devices > 0 && n_roi_ready_devices == n_connected_devices &&
				    n_in_frame >= ROI_FALLBACK_MIN_PREDICTIONS) {
				/* xmin/ymin/xmax/ymax already carry per-LED pad; no extra global pad needed. */
				int rx = (int)floorf(xmin);
				int ry = (int)floorf(ymin);
				int rw = (int)ceilf(xmax - xmin) + 1;
				int rh = (int)ceilf(ymax - ymin) + 1;
				blobwatch_process_roi_lowthresh(cam->bw, view->vframe, frame_exposure, ct->ctrl_gain,
				                                rx, ry, rw, rh, 4, 3, &view->bwobs);
				used_roi = true;
			}
		}
		if (!used_roi) {
			blobwatch_process(cam->bw, view->vframe, frame_exposure, ct->ctrl_gain, &view->bwobs);
		}

		if (view->bwobs == NULL) {
			cam->last_num_blobs = 0;
			continue;
		}

		blobservation *bwobs = view->bwobs;
		cam->last_num_blobs = bwobs->num_blobs;

		CT_TRACE(ct, "frame %" PRIu64 " TS %" PRIu64 " cam %d ROI %d,%d w/h %d,%d Blobs: %d",
		         xf->source_sequence, xf->timestamp, i, cam->roi.offset.w, cam->roi.offset.h, cam->roi.extent.w,
		         cam->roi.extent.h, bwobs->num_blobs);
	}
	uint64_t blob_extract_finish_ts = os_monotonic_get_ns();
	ct->last_blob_analysis_ms = (blob_extract_finish_ts - fast_analysis_start_ts) / U_TIME_1MS_IN_NS;
	g2_telem_event(0, xf->timestamp, G2_TELEM_EV_TRACKER_BLOB_MS, (float)ct->last_blob_analysis_ms);

	// Ready to start processing device poses now. Make sure we have the
	// LED models and collect the best estimate of the current pose
	// for each target device
	os_mutex_lock(&ct->tracked_device_lock);
	assert(ct->num_devices <= CONSTELLATION_MAX_DEVICES);

	for (int d = 0; d < ct->num_devices; d++) {
		struct constellation_tracker_device *device = ct->devices + d;

		if (!device->have_led_model) {
			if (!constellation_tracked_device_connection_get_led_model(device->connection,
			                                                           &device->led_model)) {
				continue; // Can't do anything without the LED info
			}

			CT_INFO(ct, "Constellation Tracker: Retrieved controller LED model for device %u",
			        device->led_model.id);
			device->search_led_model = t_constellation_search_model_new(&device->led_model);
			device->have_led_model = true;
		}

		struct tracking_sample_device_state *dev_state = sample->devices + sample->n_devices;
		const struct t_estimator_prior *prior = sample->estimator_prior_supported[d] ?
		    &sample->estimator_priors[d] : NULL;
		dev_state->estimator_prior = prior;
		if (!constellation_tracked_device_connection_prior_current(device->connection, prior)) {
			continue; // old camera geometry cannot be associated or cached in a new raw world
		}

		// Prior pose for matching: the fusion's RAW estimate (no body-lock ride — the visual out-of-view
		// ride must never feed back as the matcher's prior), falling back to the device's reported pose for
		// a device that doesn't expose the raw estimate.
		struct xrt_space_relation xsr;
		if (prior) {
			xsr = prior->relation; // invalid history remains a connected cold-search device
		} else if (!constellation_tracked_device_connection_get_predicted_pose(device->connection, xf->timestamp,
		                                                                &xsr) &&
		    !constellation_tracked_device_connection_get_tracked_pose(device->connection, xf->timestamp,
		                                                              &xsr)) {
			CT_DEBUG(ct, "Failed to retrieve prior pose for device %u", device->led_model.id);
			continue; // Can't retrieve the pose: means the device was disconnected
		}

		// Apply device -> LED model pose from xsr = P_world_device + P_device_model = P_world_model
		struct xrt_pose P_xrworld_model;
		math_pose_transform(&xsr.pose, &device->led_model.P_device_model, &P_xrworld_model);

		// Incoming controller pose is in OpenXR. Flip it to OpenCV for all our operations
		pose_flip_YZ(&P_xrworld_model, &dev_state->P_world_obj_prior);

		/* Covariance-driven prior gate: set the prior-consistency tolerance from the fusion's live
		 * 1-sigma uncertainty (PRIOR_GATE_SIGMA sigmas), clamped to [MIN_*_ERROR, MAX_*_ERROR]. A
		 * single scalar sigma applied isotropically is the frame-robust choice (the fusion covariance
		 * is world-frame; these bounds are used in the matcher's frame). When the fusion isn't tracking
		 * yet, fall back to the fixed floor. This widens the gate after an optical dropout (so
		 * prior-refine accepts frames it would otherwise drop to the slow search) and keeps it tight
		 * when confident. */
		float pos_bound = MIN_POS_ERROR, rot_bound = MIN_ROT_ERROR;
		double pos_std = 0.0, rot_std = 0.0, yaw_std = 0.0, tilt_std = 0.0;
		bool tilt_trusted = false;
		/* Soft mirror-flip cost's yaw scale: the live fusion YAW-AXIS 1-sigma (the orientation error about
		 * world-up alone) when tracking, else half a turn (untracked -> the yaw term vanishes). Keyed off the
		 * yaw DoF specifically, NOT the worst-direction orientation sigma — the gravity-anchored tilt is
		 * observable and tight, so folding it into the yaw scale would inflate it and conflate two DoFs.
		 * Floored at FLIP_COST_YAW_SIGMA_MIN (see its definition) so the cost cannot over-trust an
		 * over-confident prior yaw; the live sigma widens the scale above the floor after a real dropout.
		 * The TILT scale is the live horizontal-plane 1-sigma the same way, clamped to
		 * [FLIP_COST_TILT_SIGMA_MIN, GRAVITY_TILT_TOL] (see the constants' definitions). */
		float yaw_sigma = (float)FLIP_COST_YAW_SIGMA_MAX;
		float tilt_sigma = (float)GRAVITY_TILT_TOL;
		dev_state->prior_pos_std_m = -1.0f;
		bool have_uncertainty;
		if (prior) {
			have_uncertainty = prior->valid;
			pos_std = prior->position_std_m; rot_std = prior->orientation_std_rad;
			yaw_std = prior->yaw_std_rad; tilt_std = prior->tilt_std_rad;
		} else {
			have_uncertainty = constellation_tracked_device_connection_get_pose_uncertainty(
			    device->connection, &pos_std, &rot_std, &yaw_std, &tilt_std);
		}
		if (have_uncertainty) {
			pos_bound = (float)fmin(fmax(PRIOR_GATE_SIGMA * pos_std, MIN_POS_ERROR), MAX_POS_ERROR);
			rot_bound = (float)fmin(fmax(PRIOR_GATE_SIGMA * rot_std, MIN_ROT_ERROR), MAX_ROT_ERROR);
			yaw_sigma = (float)fmin(fmax(yaw_std, FLIP_COST_YAW_SIGMA_MIN), FLIP_COST_YAW_SIGMA_MAX);
			tilt_sigma = (float)fmin(fmax(tilt_std, FLIP_COST_TILT_SIGMA_MIN), GRAVITY_TILT_TOL);
			dev_state->prior_pos_std_m = (float)pos_std;
			/* TILT is driftless (gravity-anchored): trusted whenever the fusion is tracking, even through a
			 * dropout. The soft cost's yaw scale (yaw_sigma) widens with the live yaw uncertainty, so a
			 * stale yaw self-deweights rather than needing a binary trust flag. */
			tilt_trusted = prior ? prior->gravity_valid : true;
		}
			dev_state->prior_tilt_trusted = tilt_trusted;
			dev_state->prior_position_tracked =
			    (xsr.relation_flags & XRT_SPACE_RELATION_POSITION_TRACKED_BIT) != 0;
			dev_state->prior_orientation_tracked =
			    (xsr.relation_flags & XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT) != 0;
			double last_optical_age_ms = 0.0;
		if (prior) {
			last_optical_age_ms = prior->optical_age_ms;
		} else {
			constellation_tracked_device_connection_get_last_optical_age_ms(
			    device->connection, xf->timestamp, &last_optical_age_ms);
		}
		dev_state->prior_optical_stale = last_optical_age_ms > ASSOC_STALE_PRIOR_RECOVERY_AGE_MS;
		dev_state->prior_yaw_sigma_rad = yaw_sigma;
		dev_state->prior_tilt_sigma_rad = tilt_sigma;
		dev_state->prior_pos_error.x = dev_state->prior_pos_error.y = dev_state->prior_pos_error.z =
		    pos_bound;
		dev_state->prior_rot_error.x = dev_state->prior_rot_error.y = dev_state->prior_rot_error.z =
		    rot_bound;

		dev_state->gravity_ref_valid = false;
		dev_state->gravity_ref_clean = false;
		dev_state->P_world_obj_gravity = dev_state->P_world_obj_prior;
		struct xrt_quat gravity_q = {0.0f, 0.0f, 0.0f, 1.0f};
		double gravity_excess = 1e9;
		bool have_gravity;
		if (prior) {
			have_gravity = prior->valid && prior->gravity_valid;
			gravity_q = prior->gravity_orientation; gravity_excess = prior->gravity_excess_m_s2;
		} else {
			have_gravity = constellation_tracked_device_connection_get_gravity_tilt_reference(
			    device->connection, &gravity_q, &gravity_excess);
		}
		if (have_gravity) {
			struct xrt_pose P_xrworld_gravity = P_xrworld_model;
			P_xrworld_gravity.orientation = gravity_q;
			pose_flip_YZ(&P_xrworld_gravity, &dev_state->P_world_obj_gravity);
			dev_state->gravity_ref_valid = true;
			dev_state->gravity_ref_clean = gravity_excess < ASSOC_GRAVITY_CLEAN_BAND_M_S2;
		}

		dev_state->have_last_seen_pose = device->have_last_seen_pose;
		dev_state->last_seen_pose = device->last_seen_pose;

		dev_state->dev_index = d;
		dev_state->led_model = &device->led_model;

		sample->n_devices++;
	}

	/* B5 mask repair: the SLAM controller masks follow the per-frame prediction/last-seen
	 * knowledge gathered above, for every device on every processed frame — never only on
	 * accepts (an accept this frame refreshes last_seen_pose for the next frame's push,
	 * one frame period inside the halo/staleness budget). */
	push_controller_masks(ct, sample);
	os_mutex_unlock(&ct->tracked_device_lock);

	/* Evidence-accumulating retention (H5): update each camera's world-anchored static-clutter
	 * map and write per-blob retention classes before association consumes the blobs. The
	 * device exemption projects every device's LEDs at BOTH the live prediction (regardless of
	 * tracked state — the ride/OOV prediction exists while lost) and the last optically-seen
	 * pose, so a lost-and-resting controller's region is never demoted: the cold-search
	 * fall-through to suppressed blobs is work-budget-gated and must not be relied on there. */
	for (int i = 0; i < sample->n_views; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		struct tracking_sample_frame *view = sample->views + i;
		if (view->bwobs == NULL) {
			continue;
		}
		struct xrt_vec2 exempt_px[2 * CONSTELLATION_MAX_DEVICES * MAX_OBJECT_LEDS];
		int n_exempt = 0;
		for (int d = 0; d < sample->n_devices; d++) {
			struct tracking_sample_device_state *dev_state = sample->devices + d;
			append_exempt_led_px(view, cam, dev_state->led_model, &dev_state->P_world_obj_prior,
			                     exempt_px, &n_exempt, (int)ARRAY_SIZE(exempt_px));
			if (dev_state->have_last_seen_pose) {
				append_exempt_led_px(view, cam, dev_state->led_model, &dev_state->last_seen_pose,
				                     exempt_px, &n_exempt, (int)ARRAY_SIZE(exempt_px));
			}
		}
		static_map_update(&cam->static_map, view->bwobs, &cam->camera_model, &view->P_world_cam,
		                  &view->P_cam_world, sample->timestamp, exempt_px, n_exempt);

		if (g2_telem_enabled()) {
			for (int b = 0; b < view->bwobs->num_blobs; b++) {
				const struct blob *blob = &view->bwobs->blobs[b];
				g2_telem_blob((uint8_t)i, (uint64_t)xf->timestamp, blob->x, blob->y,
				              blob->brightness, blob->retention_class, blob->static_dwell_s,
				              blob->pos_var_px2);
			}
		}
	}

	constellation_associate_covariance_frame(ct, sample);
	g2_telem_event(0, xf->timestamp, G2_TELEM_EV_TRACKER_WORK_UNITS, (float)ct->last_assoc_work_units);

	uint64_t fast_analysis_finish_ts = os_monotonic_get_ns();
	ct->last_fast_analysis_ms = (fast_analysis_finish_ts - fast_analysis_start_ts) / U_TIME_1MS_IN_NS;
	g2_telem_event(0, xf->timestamp, G2_TELEM_EV_TRACKER_FAST_MS, (float)ct->last_fast_analysis_ms);

	/* Send analysis results to debug view if needed */
	enum debug_draw_flag debug_flags = DEBUG_DRAW_FLAG_NONE;
	if (ct->debug_draw_normalise)
		debug_flags |= DEBUG_DRAW_FLAG_NORMALISE;
	if (ct->debug_draw_blob_tint)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_TINT;
	if (ct->debug_draw_blob_circles)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_CIRCLE;
	if (ct->debug_draw_blob_ids)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_IDS;
	if (ct->debug_draw_blob_unique_ids)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_UNIQUE_IDS;
	if (ct->debug_draw_leds)
		debug_flags |= DEBUG_DRAW_FLAG_LEDS;
	if (ct->debug_draw_prior_leds)
		debug_flags |= DEBUG_DRAW_FLAG_PRIOR_LEDS;
	if (ct->debug_draw_last_leds)
		debug_flags |= DEBUG_DRAW_FLAG_LAST_SEEN_LEDS;
	if (ct->debug_draw_pose_bounds)
		debug_flags |= DEBUG_DRAW_FLAG_POSE_BOUNDS;
	if (ct->debug_draw_device_bounds)
		debug_flags |= DEBUG_DRAW_FLAG_DEVICE_BOUNDS;

	for (int i = 0; i < sample->n_views; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		struct tracking_sample_frame *view = sample->views + i;

		cam->debug_last_pose = view->P_world_cam;
		cam->debug_last_gravity_vector = view->cam_gravity_vector;

		if (u_sink_debug_is_active(&cam->debug_sink)) {
			struct xrt_frame *xf_src = view->vframe;
			struct xrt_frame *xf_dbg = NULL;

			u_frame_create_one_off(XRT_FORMAT_R8G8B8, xf_src->width, xf_src->height, &xf_dbg);
			xf_dbg->timestamp = xf_src->timestamp;

			debug_draw_blobs_leds(xf_dbg, xf_src, debug_flags, view, i, &cam->camera_model, sample->devices,
			                      sample->n_devices);

			u_sink_debug_push_frame(&cam->debug_sink, xf_dbg);
			xrt_frame_reference(&xf_dbg, NULL);
		}
	}

	atomic_fetch_add_explicit(&ct->frames_completed, 1, memory_order_release);
	constellation_tracking_sample_free(sample);
}

static void
constellation_tracker_node_destroy(struct xrt_frame_node *node)
{
	struct t_constellation_tracker *ct = container_of(node, struct t_constellation_tracker, node);

	DRV_TRACE_MARKER();
	CT_DEBUG(ct, "Destroying constellation tracker");

	// Unlink the device connections and release the models
	os_mutex_lock(&ct->tracked_device_lock);
	for (int i = 0; i < ct->num_devices; i++) {
		struct constellation_tracker_device *device = ct->devices + i;

		// Clean up the LED tracking model
		t_constellation_led_model_clear(&device->led_model);
		if (device->search_led_model) {
			t_constellation_search_model_free(device->search_led_model);
		}

		if (device->connection != NULL) {
			t_constellation_tracked_device_connection_disconnect(device->connection);
			device->connection = NULL;
		}
	}
	os_mutex_unlock(&ct->tracked_device_lock);
	os_mutex_destroy(&ct->tracked_device_lock);

	u_worker_group_reference(&ct->cold_search_group, NULL);
	u_worker_thread_pool_reference(&ct->cold_search_pool, NULL);

	//! Clean up
	for (int i = 0; i < ct->cam_count; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;

		u_sink_debug_destroy(&cam->debug_sink);

		for (int dev = 0; dev < CONSTELLATION_MAX_DEVICES; dev++) {
			if (cam->cold_search[dev]) {
				correspondence_search_free(cam->cold_search[dev]);
			}
		}

		if (cam->bw) {
			blobwatch_free(cam->bw);
		}
	}

	u_var_remove_root(ct);
	free(ct);
}

int
t_constellation_tracker_create(struct xrt_frame_context *xfctx,
                               struct xrt_device *hmd_xdev,
                               struct t_constellation_camera_group *cams,
                               struct t_constellation_tracker **out_tracker,
                               struct xrt_frame_sink **out_sink,
                               struct xrt_device_masks_sink *controller_mask_sink)
{
	DRV_TRACE_MARKER();

	int ret;
	struct t_constellation_tracker *ct = calloc(1, sizeof(struct t_constellation_tracker));
	if (ct == NULL) {
		return -1;
	}

	ct->log_level = debug_get_log_option_ct_log();

	/* Init the lock before any other resource: every later failure path unwinds through
	 * constellation_tracker_node_destroy, which locks and then destroys this mutex. */
	ret = os_mutex_init(&ct->tracked_device_lock);
	if (ret != 0) {
		CT_ERROR(ct, "Failed to init tracked device mutex!");
		free(ct);
		return -1;
	}

	ct->debug_draw_blob_tint = true;
	ct->debug_draw_blob_ids = true;
	ct->hmd_xdev = hmd_xdev;
	ct->controller_masks_sink = controller_mask_sink;
	atomic_store_explicit(&ct->frames_completed, 0, memory_order_relaxed);
	ct->cold_search_pool = u_worker_thread_pool_create(ASSOC_COLD_SEARCH_STARTING_WORKERS,
	                                                   ASSOC_COLD_SEARCH_THREADS, "G2 cold search");
	if (ct->cold_search_pool != NULL) {
		ct->cold_search_group = u_worker_group_create(ct->cold_search_pool);
	}

	// Set up the per-camera constellation tracking pieces config and pose
	ct->cam_count = cams->cam_count;
	ct->ctrl_gain = cams->ctrl_gain;

	/* Re-denominate the DN-based clutter-veto prongs at the commanded gain (see the definitions of
	 * assoc_single_view_clutter_* for the derivation). Exact no-ops at gain 16 / unknown (m = 1). */
	{
		const float m = blobwatch_gain_multiplier(ct->ctrl_gain);
		const float k_m = blobwatch_dim_noise_k(ct->ctrl_gain);
		const float k_16 = blobwatch_dim_noise_k(0);
		assoc_single_view_clutter_max_brightness = ASSOC_SINGLE_VIEW_CLUTTER_MAX_BRIGHTNESS * m;
		const float rk_m = k_m / assoc_single_view_clutter_max_brightness;
		const float rk_16 = k_16 / ASSOC_SINGLE_VIEW_CLUTTER_MAX_BRIGHTNESS;
		assoc_single_view_clutter_min_var_px2 =
		    ASSOC_SINGLE_VIEW_CLUTTER_MIN_VAR_PX2 * (1.0f + rk_m * rk_m) / (1.0f + rk_16 * rk_16);
	}

	for (int i = 0; i < ct->cam_count; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		struct t_constellation_camera *cam_cfg = cams->cams + i;

		cam->roi = cam_cfg->roi;
		cam->P_imu_cam = cam_cfg->P_imu_cam;
		cam->slam_tracking_index = cam_cfg->slam_tracking_index;

		/* Init the camera model with size and distortion */
		cam->camera_model.width = cam_cfg->roi.extent.w;
		cam->camera_model.height = cam_cfg->roi.extent.h;
		t_camera_model_params_from_t_camera_calibration(&cam_cfg->calibration, &cam->camera_model.calib);

		cam->bw = blobwatch_new(cam_cfg->blob_min_threshold, (uint8_t)i);
		if (cam->bw == NULL) {
			CT_ERROR(ct, "Failed to allocate blobwatch for camera %d!", i);
			constellation_tracker_node_destroy(&ct->node);
			return -1;
		}
		for (int dev = 0; dev < CONSTELLATION_MAX_DEVICES; dev++) {
			cam->cold_search[dev] = correspondence_search_new(&cam->camera_model);
			if (cam->cold_search[dev] == NULL) {
				CT_ERROR(ct, "Failed to allocate correspondence search for camera %d!", i);
				constellation_tracker_node_destroy(&ct->node);
				return -1;
			}
		}
	}

	// Set up frame receiver
	ct->base.push_frame = constellation_tracker_receive_frame;

	// Setup node
	struct xrt_frame_node *xfn = &ct->node;
	xfn->break_apart = constellation_tracker_node_break_apart;
	xfn->destroy = constellation_tracker_node_destroy;

	// Fast processing thread
	ct->fast_process_sink.push_frame = constellation_tracker_process_frame_fast;

	if (!u_sink_queue_create(xfctx, MAX_FAST_QUEUE_SIZE, &ct->fast_process_sink, &ct->fast_q_sink)) {
		CT_ERROR(ct, "Failed to init fast analysis queue!");
		constellation_tracker_node_destroy(&ct->node);
		return -1;
	}

	u_var_add_root(ct, "Constellation Tracker", false);
	u_var_add_log_level(ct, &ct->log_level, "Log Level");
	u_var_add_ro_i32(ct, &ct->num_devices, "Num Devices");
	u_var_add_ro_u64(ct, &ct->last_frame_timestamp, "Last Frame Timestamp");
	u_var_add_ro_u64(ct, &ct->last_blob_analysis_ms, "Blob tracking time (ms)");
	u_var_add_ro_u64(ct, &ct->last_fast_analysis_ms, "Fast analysis time (ms)");
	u_var_add_ro_u64(ct, &ct->last_assoc_work_units, "Cold search work units");

	u_var_add_bool(ct, &ct->debug_draw_normalise, "Debug: Normalise source frame");
	u_var_add_bool(ct, &ct->debug_draw_blob_tint, "Debug: Tint blobs by device assignment");
	u_var_add_bool(ct, &ct->debug_draw_blob_circles, "Debug: Draw circles around blobs");
	u_var_add_bool(ct, &ct->debug_draw_blob_ids, "Debug: Draw LED id labels for blobs");
	u_var_add_bool(ct, &ct->debug_draw_blob_unique_ids, "Debug: Draw blobs tracking ID");
	u_var_add_bool(ct, &ct->debug_draw_leds, "Debug: Draw LED position markers for found poses");
	u_var_add_bool(ct, &ct->debug_draw_prior_leds, "Debug: Draw LED markers for prior poses");
	u_var_add_bool(ct, &ct->debug_draw_last_leds, "Debug: Draw LED markers for last observed poses");
	u_var_add_bool(ct, &ct->debug_draw_pose_bounds, "Debug: Draw LED bounds rect for found poses");
	u_var_add_bool(ct, &ct->debug_draw_device_bounds, "Debug: Draw device bounds rect for found poses");

	for (int i = 0; i < ct->cam_count; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		u_var_add_ro_i32(ct, &cam->last_num_blobs, "Num Blobs");
		u_var_add_pose(ct, &cam->debug_last_pose, "Last view pose");
		u_var_add_vec3_f32(ct, &cam->debug_last_gravity_vector, "Last gravity vector");

		char cam_name[64];
		snprintf(cam_name, sizeof(cam_name), "Cam %u", i);
		u_sink_debug_init(&cam->debug_sink);
		u_var_add_sink_debug(ct, &cam->debug_sink, cam_name);
	}

	// Hand ownership to the frame context
	xrt_frame_context_add(xfctx, &ct->node);

	CT_DEBUG(ct, "Constellation tracker created");

	*out_tracker = ct;
	*out_sink = &ct->base;

	return 0;
}

static void
constellation_tracked_device_connection_destroy(struct t_constellation_tracked_device_connection *ctdc)
{
	DRV_TRACE_MARKER();

	os_mutex_destroy(&ctdc->lock);
	free(ctdc);
}

static struct t_constellation_tracked_device_connection *
constellation_tracked_device_connection_create(int id,
                                               struct xrt_device *xdev,
                                               struct t_constellation_tracked_device_callbacks *cb,
                                               struct t_constellation_tracker *tracker)
{
	DRV_TRACE_MARKER();

	assert(xdev != NULL);
	assert(cb != NULL);

	struct t_constellation_tracked_device_connection *ctdc =
	    calloc(1, sizeof(struct t_constellation_tracked_device_connection));
	if (ctdc == NULL) {
		return NULL;
	}

	ctdc->id = id;
	ctdc->xdev = xdev;
	ctdc->cb = cb;
	ctdc->tracker = tracker;

	/* Init 2 references - one for the tracked device, one for the tracker */
	xrt_reference_inc(&ctdc->ref);
	xrt_reference_inc(&ctdc->ref);

	int ret = os_mutex_init(&ctdc->lock);
	if (ret != 0) {
		CT_ERROR(tracker, "Constellation tracker device connection: Failed to init mutex!");
		/* Not connection_destroy(): that would destroy the never-initialized mutex. */
		free(ctdc);
		return NULL;
	}

	return ctdc;
}

struct t_constellation_tracked_device_connection *
t_constellation_tracker_add_device(struct t_constellation_tracker *ct,
                                   struct xrt_device *xdev,
                                   struct t_constellation_tracked_device_callbacks *cb)
{
	os_mutex_lock(&ct->tracked_device_lock);
	assert(ct->num_devices < CONSTELLATION_MAX_DEVICES);

	CT_DEBUG(ct, "Constellation tracker: Adding device %d", ct->num_devices);

	struct t_constellation_tracked_device_connection *ctdc =
	    constellation_tracked_device_connection_create(ct->num_devices, xdev, cb, ct);
	if (ctdc != NULL) {
		struct constellation_tracker_device *device = ct->devices + ct->num_devices;
		device->connection = ctdc;
		device->last_matched_cam = -1;
		ct->num_devices++;

		const char *device_type;
		switch (xdev->device_type) {
		case XRT_DEVICE_TYPE_HMD: device_type = "HMD"; break;
		case XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER: device_type = "Right"; break;
		case XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER: device_type = "Left"; break;
		case XRT_DEVICE_TYPE_ANY_HAND_CONTROLLER: device_type = "Any"; break;
		case XRT_DEVICE_TYPE_GENERIC_TRACKER: device_type = "Tracker"; break;
		default: device_type = "Unknown"; break;
		}

		char dev_name[64];
		snprintf(dev_name, sizeof(dev_name), "Device %u - %s", ct->num_devices, device_type);
		u_var_add_ro_text(ct, "Device", dev_name);
		u_var_add_pose(ct, &device->last_seen_pose, "Last observed global pose");
		u_var_add_u64(ct, &device->last_seen_pose_ts, "Last observed pose");
		u_var_add_ro_i32(ct, &device->last_matched_blobs, "Last matched Blobs");
		u_var_add_ro_i32(ct, &device->last_matched_cam, "Last observed camera #");
		u_var_add_pose(ct, &device->last_matched_cam_pose, "Last observed camera pose");
	}

	os_mutex_unlock(&ct->tracked_device_lock);
	return ctdc;
}

void
t_constellation_tracked_device_connection_disconnect(struct t_constellation_tracked_device_connection *ctdc)
{
	os_mutex_lock(&ctdc->lock);
	ctdc->disconnected = true;
	os_mutex_unlock(&ctdc->lock);

	if (xrt_reference_dec_and_is_zero(&ctdc->ref)) {
		constellation_tracked_device_connection_destroy(ctdc);
	}
}

uint64_t
t_constellation_tracker_debug_frames_completed(struct t_constellation_tracker *ct)
{
	return atomic_load_explicit(&ct->frames_completed, memory_order_acquire);
}
