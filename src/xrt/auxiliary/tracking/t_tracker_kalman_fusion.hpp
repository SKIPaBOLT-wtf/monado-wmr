// Copyright 2019, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  IMU + optical (constellation) ESKF fusion interface.
 * @author Pete Black <pblack@collabora.com>
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @author Rylie Pavlik <rylie.pavlik@collabora.com>
 * @ingroup aux_tracking
 */

#pragma once

#ifndef __cplusplus
#error "This header is C++-only."
#endif

#include "xrt/xrt_defines.h"
#include "xrt/xrt_tracking.h"

#include "util/u_time.h"

#include <memory>
#include <vector>

// The per-LED observation + camera-view structs are defined once as C structs (so the C driver code
// can build them) and aliased here for the C++ filter — single definition, no duplication.
#include "t_tracker_kalman_fusion_c.h"


namespace xrt::auxiliary::tracking {

//! @see kalman_led_observation (C definition). Pixel is undistorted-normalized; led_obj is in the
//! object frame with any OpenXR<->OpenCV flip pre-applied by the caller.
using LEDObservation = ::kalman_led_observation;
//! @see kalman_led_camera_view. fx=fy=1/cx=cy=0 for normalized obs; extrinsic maps the filter's
//! world frame to the camera frame (caller folds in the YZ flip).
using LEDCameraView = ::kalman_led_camera_view;

bool
predict_led_gate_from_prior(const t_estimator_prior &prior, const LEDObservation &obs,
                           const LEDCameraView &view, float out_zhat[2], float out_S[4]);

class KalmanFusionInterface
{
public:
	static std::unique_ptr<KalmanFusionInterface>
	create();
	virtual ~KalmanFusionInterface() = default;

	virtual bool
	get_estimator_prior(timepoint_ns when_ns, t_estimator_prior *out_prior) = 0;
	virtual uint64_t
	get_world_generation() = 0;


	virtual void
	add_ui(void *root, const char *device_name) = 0;

	/*!
	 * @brief If you've lost sight of the position tracking and won't even
	 * enter another function in this class.
	 */
	virtual void
	clear_position_tracked_flag() = 0;

	virtual void
	process_imu_data(const struct xrt_imu_sample *sample,
	                 const struct xrt_vec3 *accel_variance_optional,
	                 const struct xrt_vec3 *gyro_variance_optional) = 0;
	//! Hardware status without an IMU measurement; may confirm a recently established stationary state.
	virtual void
	process_idle_status(timepoint_ns timestamp_ns) = 0;
	virtual void
	process_pose(const struct xrt_pose_sample *sample,
	             const struct xrt_vec3 *position_variance_optional,
	             const struct xrt_vec3 *orientation_variance_optional,
	             const float residual_limit,
	             const struct xrt_pose *hmd_world_pose) = 0;

	/*!
	 * Position-only optical observation for visually-supported but orientation-ambiguous frames. This lets the
	 * front-end use a controller whose LEDs constrain translation while mirror-twin/yaw evidence is not strong
	 * enough to accept a full 6DoF pose. It never changes orientation directly.
	 */
	virtual void
	process_position(const timepoint_ns timestamp_ns,
	                 const struct xrt_vec3 *position,
	                 const struct xrt_vec3 *position_variance_optional,
	                 const struct xrt_pose *hmd_world_pose,
	                 bool refresh_optical_anchor) = 0;

	/*!
	 * Cache-only PnP candidate for per-LED divergence recovery. This updates only the guarded re-anchor
	 * target used by process_led_observations; it does not adopt the pose, change optical freshness, or
	 * report tracking.
	 */
	virtual void
	cache_pnp_pose_candidate(const timepoint_ns timestamp_ns,
	                         const struct xrt_pose *pose,
	                         const struct xrt_pose *hmd_world_pose) = 0;

	/*!
	 * Tightly-coupled optical update: fold each matched LED's reprojection
	 * into the filter directly, instead of solving one PnP pose and feeding
	 * that. Works with as few as a single LED, so frames that have too few
	 * matched LEDs for a PnP pose still inform the filter. Each LED is gated
	 * individually (a per-LED pixel innovation threshold) so a mislabelled
	 * correspondence cannot drag the pose.
	 *
	 * @param timestamp_ns    capture time of this view.
	 * @param obs             matched (blob pixel, LED object-frame point) pairs.
	 * @param view            pinhole intrinsics + world->camera extrinsic.
	 * @param pixel_variance  per-axis pixel measurement variance (px^2).
	 * @param max_innov_px    per-LED robust gate: skip an LED whose pixel
	 *                        innovation magnitude exceeds this (mislabel guard).
	 * @param feed            true: fold the observations into the filter (tightly-coupled
	 *                        update). false: DIAGNOSTIC only — compute the per-LED reprojection
	 *                        residual against the current pose without touching the filter
	 *                        (read-only; used to verify the frame/undistort transforms before
	 *                        enabling the feed).
	 * @param hmd_world_pose  live HMD pose in the controller's world frame (or null). Used to gate
	 *                        adoption on the controller-to-HMD distance (arm reach, room-roam-invariant)
	 *                        and to capture the body-lock offset when a fold constrains position.
	 * @return feed=true: the number of LEDs actually folded this frame (after the per-LED gate; 0 =
	 *         all gated out as mislabels), or -1 if the filter is not yet tracking (awaiting a
	 *         process_pose bootstrap). feed=false (diagnostic): the RMS per-LED reprojection residual
	 *         in NORMALIZED image units (~residual_px / focal; < ~0.01 ⇒ frame transform correct), or
	 *         -1 if no pose was available.
	 */
	virtual float
	process_led_observations(const timepoint_ns timestamp_ns,
	                         const std::vector<LEDObservation> &obs,
	                         const LEDCameraView &view,
	                         const struct xrt_vec2 *pixel_variance,
	                         const float max_innov_px,
	                         const bool feed,
	                         const struct xrt_pose *hmd_world_pose) = 0;

	/*!
	 * @param hmd_world_pose live HMD pose in the controller's world frame (or null). When position is no
	 * longer optically observable, the reported controller rides this live HMD pose at its last
	 * controller-to-HMD offset (body-lock), instead of dead-reckoning away in world frame. Null falls
	 * back to the world-frame hold at the last optically-observed position.
	 */
	virtual void
	get_prediction(const timepoint_ns when_ns,
	               struct xrt_space_relation *out_relation,
	               const struct xrt_pose *hmd_world_pose,
	               struct xrt_pose *out_from_raw = nullptr,
	               uint64_t *out_generation = nullptr,
	               bool hmd_pose_is_presented = false) = 0;

	/*!
	 * Raw predicted pose for a CONSUMER OF THE ESTIMATE (the constellation matcher's prior), as opposed to
	 * get_prediction's compositor REPORT: the honest filter belief with no body-lock / reach / re-entry
	 * transforms, so feeding it back as the matcher prior never lets the visual ride mislead the front-end.
	 * Identity / untracked flags until the filter is tracking.
	 */
	virtual void
	get_predicted_pose(const timepoint_ns when_ns, struct xrt_space_relation *out_relation) = 0;

	/*!
	 * Predict one LED's image point and its 2x2 innovation covariance S = H P H^T + R for the
	 * covariance-driven associator (the per-LED gate ellipse). Uses the EXACT same H/projection as the
	 * tightly-coupled fold (single source of truth), so the gate the front-end sizes matches the filter's
	 * own measurement model. @p out_zhat receives the predicted pixel; @p out_S receives S row-major
	 * (S00,S01,S10,S11). Returns false (and writes nothing) until the filter is tracking.
	 */
	virtual bool
	predict_led_gate(const timepoint_ns when_ns,
	                 const LEDObservation &obs,
	                 const LEDCameraView &view,
	                 float out_zhat[2],
	                 float out_S[4])
	{
		(void)when_ns;
		(void)obs;
		(void)view;
		(void)out_zhat;
		(void)out_S;
		return false;
	}

	/*!
	 * Diagnostics / tests: the raw per-LED reprojection Jacobian at the current state — @p out_H row-major
	 * 2x6 [d/dp(3), d/dtheta_world(3)] from the SAME led_project_jacobian the fold/gate use, plus the
	 * predicted pixel @p out_zhat. Lets a test pin the H convention directly against a finite-difference
	 * (the global-vs-local orientation-Jacobian regression), which S = H P H^T + R cannot isolate when
	 * P_theta is isotropic. Off the hot path. Returns false if not tracking.
	 */
	virtual bool
	debug_predict_led_jacobian(const LEDObservation &obs, const LEDCameraView &view, double out_zhat[2],
	                           double out_H[12])
	{
		(void)obs;
		(void)view;
		(void)out_zhat;
		(void)out_H;
		return false;
	}

	/*!
	 * Diagnostics / tests: filter consistency. Writes the mean per-DOF NIS over the last folds (a
	 * correctly-tuned filter averages ~1) and the most recent fold's per-DOF NIS, and returns the number
	 * of folds in the window (0 if none yet). Either out pointer may be null.
	 */
	virtual int
	debug_get_nis_stats(double *mean_per_dof, double *last_per_dof)
	{
		(void)mean_per_dof;
		(void)last_per_dof;
		return 0;
	}

	/*!
	 * Current 1-sigma uncertainty of the predicted pose. position_std (m) and orientation_std (rad) are
	 * the worst-direction / largest-eigenvalue stds, applied isotropically — frame-conservative, since the
	 * covariance is world-frame while a consumer's gate may be in another frame, and variance in any
	 * direction is bounded by the largest eigenvalue; used to size the prior-consistency gate (tight when
	 * well-tracked, wide right after an optical dropout). yaw_std (rad), when non-null, is the orientation
	 * std about world-up ALONE — the uncertain yaw DoF, separated from the gravity-anchored (observable)
	 * tilt — for the mirror-flip cost's yaw scale, which must be sharp when yaw is confident. tilt_std
	 * (rad), when non-null, is the worst-direction std within the horizontal plane — the gravity-observable
	 * tilt DoFs alone — for the prior's tilt scale, tight when gravity-observed and honestly widened during
	 * dynamics. Any out pointer may be null. Wait-free (reads the published snapshot). Returns false if not
	 * tracking.
	 */
	virtual bool
	get_pose_uncertainty(double *position_std, double *orientation_std, double *yaw_std, double *tilt_std)
	{
		(void)position_std;
		(void)orientation_std;
		(void)yaw_std;
		(void)tilt_std;
		return false;
	}

	/*!
	 * Gravity-tilt-corrected held orientation (world<-body): current orientation with tilt snapped to
	 * accelerometer gravity and yaw preserved. @p out_excess_m_s2 is |||f|| - g|; small means low linear
	 * acceleration. Returns false until tracking and a finite IMU gravity reference exist.
	 */
	virtual bool
	get_gravity_tilt_reference(struct xrt_quat *out_gravity_corrected_q, double *out_excess_m_s2)
	{
		(void)out_gravity_corrected_q;
		(void)out_excess_m_s2;
		return false;
	}

	/*!
	 * Diagnostics / tests: copy the current estimate's 3x3 position covariance
	 * (row-major, m^2). Off the hot path. Returns false if unavailable or not yet
	 * tracking. Used by the filter-consistency (NEES) test and live debugging.
	 */
	virtual bool
	debug_get_position_covariance(double cov_row_major[9])
	{
		(void)cov_row_major;
		return false;
	}

	/*!
	 * Diagnostics / tests: copy the 6x6 pose covariance (row-major) ordered [position(3), global
	 * orientation(3)] — the [EP,ET] sub-blocks of the 15x15 P. Lets a test reconstruct the per-LED
	 * innovation covariance S = H P H^T + R from the same blocks predict_led_gate uses. Off the hot
	 * path. Returns false if not tracking.
	 */
	virtual bool
	debug_get_pose_covariance(double cov6_row_major[36])
	{
		(void)cov6_row_major;
		return false;
	}

	/*!
	 * Diagnostics / tests: copy the full 15x15 error-state covariance (row-major), ordered
	 * [position(3), velocity(3), global orientation(3), accel bias(3), gyro bias(3)]. Lets a test
	 * assert P's structural health (positive semi-definiteness, cleared cross-covariance blocks)
	 * after coast/recovery paths. Off the hot path. Returns false if not tracking.
	 */
	virtual bool
	debug_get_state_covariance(double cov15_row_major[225])
	{
		(void)cov15_row_major;
		return false;
	}

	//! Diagnostics / tests: the current online accel-scale correction (nominal 1.0). 1.0 if unsupported.
	virtual double
	debug_get_accel_scale()
	{
		return 1.0;
	}

	//! Diagnostics / tests: the fitted full accel calibration matrix (row-major, per-axis scale +
	//! misalignment). Returns false until the ellipsoid has been fitted from enough rest orientations.
	virtual bool
	debug_get_accel_calibration(double T_row_major[9])
	{
		(void)T_row_major;
		return false;
	}

	/*!
	 * Cross-session IMU calibration cache. The driver persists the converged per-controller gyro/accel
	 * bias + accel scale (keyed by serial) and seeds a new session with them, so the first second is
	 * accurate before any stance and a large physical bias survives. set_* applies a prior (before
	 * tracking); get_* reads the current converged values to persist (returns false until a calibrated
	 * stance has occurred, i.e. the estimate is trustworthy).
	 */
	virtual void
	set_imu_calibration(const double gyro_bias[3], const double accel_bias[3], double accel_scale)
	{
		(void)gyro_bias;
		(void)accel_bias;
		(void)accel_scale;
	}
	virtual bool
	get_imu_calibration(double gyro_bias[3], double accel_bias[3], double *accel_scale)
	{
		(void)gyro_bias;
		(void)accel_bias;
		(void)accel_scale;
		return false;
	}

	/*!
	 * Optically-derived IMU INTRINSICS (computed offline by imu_calib_from_optical.py and persisted per
	 * controller serial alongside the bias cache). Two 3x3 corrections applied in the IMU integration
	 * BEFORE bias/integration drift the state:
	 *  - @p gyro_correction (M_g, row-major): corrected body rate = M_g * (gyro_meas - bg). Captures gyro
	 *    scale (its singular values) AND the gyro->device misalignment (its orthogonal part) in one matrix
	 *    — the exact M the offline tool fits from phi_optical = M * gyro_integral (single source of truth).
	 *  - @p accel_correction (T_a, row-major): corrected specific force = T_a * (accel_meas - ba) — the
	 *    accel ellipsoid (per-axis scale + cross-axis). Seeds m_accel_T so the offline ellipsoid is applied
	 *    immediately, rather than waiting for the rarely-occurring online rest-orientation spread.
	 * Either may be identity (then that channel is uncorrected — no regression). set_* applies a prior
	 * before tracking; get_* reads the currently-applied intrinsics to persist. get_* returns false if
	 * neither channel is a real correction (nothing worth persisting). The bias/scale path is unchanged.
	 */
	virtual void
	set_imu_intrinsics(const double gyro_correction[9], const double accel_correction[9])
	{
		(void)gyro_correction;
		(void)accel_correction;
	}
	virtual bool
	get_imu_intrinsics(double gyro_correction[9], double accel_correction[9])
	{
		(void)gyro_correction;
		(void)accel_correction;
		return false;
	}

	/*!
	 * Out-of-view body-plausibility update against the live head pose @p hmd_pose. Out of view, the fusion
	 * softly anchors the controller position to its last body-relative (rigid HMD-relative) point so the
	 * dead-reckon drift stays bounded. The driver calls this per controller IMU sample; a no-op while the
	 * controller is in view.
	 */
	virtual void
	update_body_anchor(const struct xrt_pose *hmd_pose)
	{
		(void)hmd_pose;
	}

	/*!
	 * World re-anchor: the head tracker's world frame moved by the rigid transform @p delta
	 * (x' = delta.q * x + delta.p, world coords) in one detected step — a SLAM relocalization /
	 * reset (the B2 head resnap guard's detector, pivoted at the pre-step head position so
	 * head-relative geometry is preserved). The filter's world-frame state — mean, covariance
	 * orientation, cached world points/vectors and the out-of-sequence checkpoint — is
	 * transformed by delta so the prior lands in the NEW world and the next optical accepts
	 * continue seamlessly, instead of disagreeing with the covariance gate by the full jump.
	 * A frame-exact linear transform: body-frame state (IMU biases, intrinsics) and
	 * head-relative logic are untouched, and delta == identity is a no-op.
	 */
	virtual void
	re_anchor_world_with_presentation(const struct xrt_pose *delta, const struct xrt_vec3 *new_raw_pivot,
	                                  timepoint_ns publication_ns, double gyro_dps, double speed_mps) = 0;

	virtual void
	re_anchor_world(const struct xrt_pose *delta)
	{
		(void)delta;
	}

	/*!
	 * Diagnostics / tests: the named regime of the last get_prediction report, as a stable integer code so a
	 * test can assert FSM<->flags parity without the implementation enum. Codes:
	 * 0 Invalid, 1 VisualAccuracy, 2 InertialFastMotion, 3 WorldLocked, 4 BodyLocked, 5 ConfusedPosition.
	 * Optionally also writes the name to @p name_out (capacity @p name_cap). Off the hot path.
	 */
	virtual int
	debug_get_fusion_state(char *name_out, size_t name_cap)
	{
		(void)name_out;
		(void)name_cap;
		return 0;
	}

	virtual bool
	debug_get_last_optical_age_ms(timepoint_ns when_ns, double *age_ms)
	{
		(void)when_ns;
		(void)age_ms;
		return false;
	}

	virtual bool
	debug_get_oov_report(timepoint_ns when_ns,
	                     const struct xrt_pose *hmd_world_pose,
	                     struct kalman_fusion_oov_debug *out_debug)
	{
		(void)when_ns;
		(void)hmd_world_pose;
		(void)out_debug;
		return false;
	}
};
} // namespace xrt::auxiliary::tracking
