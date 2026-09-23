// Copyright 2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Implementation of LED constellation tracking logic
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "os/os_threading.h"
#include "tracking/t_tracking.h"
#include "tracking/t_estimator_prior.h"
#include "tracking/t_led_models.h"
#include "util/u_sink.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_frame.h"
#include "xrt/xrt_tracking.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @defgroup constellation LED constellation tracking
 * @ingroup tracking
 *
 * @brief Tracker for devices with LED constellations
 */

/*!
 * @dir tracking/constellation
 *
 * @brief @ref constellation tracking files.
 */

struct t_constellation_tracker;
struct t_constellation_tracked_device_connection;

struct t_constellation_camera
{
	//!< IMU to camera pose
	struct xrt_pose P_imu_cam;
	//! ROI in the full frame mosaic
	struct xrt_rect roi;
	//! Intrinsics and distortion parameters
	struct t_camera_calibration calibration;
	//! Minimum blob brightness threshold
	uint8_t min_threshold;
	//! Minimum blob brightness threshold for pixel inclusion
	uint8_t blob_min_threshold;
	//! The index into the slam tracking camera array this camera represents
	size_t slam_tracking_index;
};

struct t_constellation_camera_group
{
	int cam_count; //!< Number of cameras
	/*! Commanded analog-gain register for the controller-tracking (short LED exposure) slot, shared by
	 * all cameras (one operating point per session). 1/16 fine-gain units, valid 16..255; 0 = unknown
	 * (drivers/recordings that predate gain plumbing) and is treated as the gain-16 calibration point,
	 * so old-capture replays are bit-identical. The tracker's DN-denominated constants are scaled by
	 * the resulting brightness multiplier m = gain/16 (see blobwatch_dim_noise_k). */
	uint16_t ctrl_gain;
	struct t_constellation_camera cams[XRT_TRACKING_MAX_SLAM_CAMS];
};

int
t_constellation_tracker_create(struct xrt_frame_context *xfctx,
                               struct xrt_device *hmd_xdev,
                               struct t_constellation_camera_group *cams,
                               struct t_constellation_tracker **out_tracker,
                               struct xrt_frame_sink **out_sink,
                               struct xrt_device_masks_sink *controller_mask_sink);

/*!
 * Serialise a camera group (per-camera intrinsics, distortion model + params, IMU->camera extrinsic,
 * mosaic ROI, blob thresholds and the commanded controller-slot gain) to JSON. Lets the exact
 * calibration the tracker is built from be
 * persisted for OFFLINE replay of the full controller VIO against recorded raw frames — the per-unit
 * camera calibration otherwise lives only in headset flash. Self-describing ("g2-constellation-cameras"
 * v1). No-op on NULL args.
 */
void
t_constellation_camera_group_dump_json(const struct t_constellation_camera_group *cams, FILE *f);

/*!
 * One matched constellation LED for the tightly-coupled per-LED fusion feed: the observed blob
 * centroid (UNDISTORTED, normalized image coords — i.e. already through t_camera_models_undistort)
 * paired with the LED's 3D model position in the device's OpenXR object frame (the OpenXR<->OpenCV
 * YZ flip vs the constellation's OpenCV frame is pre-applied by the tracker).
 */
struct t_constellation_led_obs
{
	struct xrt_vec2 obs_px;  //!< undistorted blob centroid, in camera PIXELS (distortion removed)
	struct xrt_vec3 led_obj; //!< LED position in the OpenXR object frame (m)
	float pos_var_px2;      //!< optional isotropic centroid variance; <= 0 uses the fusion default
};

//! Camera pinhole intrinsics for one view, so the fusion can reproject in physical pixels (the per-LED
//! measurement noise/gate are then focal-independent). Matches @ref t_constellation_led_obs::obs_px,
//! which is the undistorted pixel position under exactly these intrinsics.
struct t_constellation_cam_calib
{
	float fx, fy, cx, cy;
};

struct t_constellation_tracked_device_callbacks
{
	//! Capture one immutable prior before ROI/association. True means this interface is supported;
	//! out->valid=false means cold search, NOT disconnected or a rendered-pose fallback.
	bool (*get_estimator_prior)(struct xrt_device *xdev, timepoint_ns when_ns, struct t_estimator_prior *out);
	//! Called under the SAME connection lock as the ensuing observation callback and world reanchor.
	//! The token belongs to this device, including when history is missing at cold start.
	bool (*validate_prior_epoch)(struct xrt_device *xdev, const struct t_estimator_prior *prior);
	//! Pure projection from the frozen bundle and this exposure's frozen camera geometry.
	bool (*predict_led_gate_from_prior)(const struct t_estimator_prior *prior,
	    const struct xrt_pose *P_xrworld_cam, const struct t_constellation_cam_calib *calib,
	    const struct xrt_vec3 *led_obj, float out_zhat[2], float out_S[4]);

	bool (*get_led_model)(struct xrt_device *xdev, struct t_constellation_led_model *led_model);
	void (*notify_frame_received)(struct xrt_device *xdev, uint64_t frame_mono_ns, uint64_t frame_sequence);
	void (*push_observed_pose)(struct xrt_device *xdev, timepoint_ns frame_mono_ns, const struct xrt_pose *pose);
	void (*push_brightness_update)(struct xrt_device *xdev, uint8_t average_brightness);
	//! Per-view matched LED correspondences for the tightly-coupled (ESKF) fusion feed. @p
	//! P_xrworld_cam maps the device's OpenXR world frame to this camera (OpenCV camera frame),
	//! consistent with @ref t_constellation_led_obs::led_obj; @p cam_calib are the view's pinhole
	//! intrinsics matching @ref t_constellation_led_obs::obs_px. Optional — may be left NULL.
	void (*push_observed_leds)(struct xrt_device *xdev, timepoint_ns frame_mono_ns,
	                           const struct xrt_pose *P_xrworld_cam,
	                           const struct t_constellation_cam_calib *cam_calib,
	                           const struct t_constellation_led_obs *leds, size_t led_count);
	//! Current 1-sigma uncertainty of the device's predicted (prior) pose from the fusion covariance:
	//! position_std (m) and orientation_std (rad) are the worst-direction stds the matcher uses to size
	//! its prior-consistency gate (tight when tracked, wide after a dropout); yaw_std (rad), when
	//! non-null, is the orientation std about world-up ALONE — the uncertain yaw DoF, separated from the
	//! gravity-anchored (observable) tilt — for the mirror-flip cost's yaw scale; tilt_std (rad), when
	//! non-null, is the worst-direction std within the horizontal (gravity-observable) plane — for the
	//! prior's tilt scale. Any out pointer may be NULL. Optional callback — may be NULL, or return false
	//! until the fusion is tracking; the matcher then falls back to its fixed default bounds.
	bool (*get_pose_uncertainty)(struct xrt_device *xdev, double *position_std, double *orientation_std,
	                             double *yaw_std, double *tilt_std);
	//! Gravity-tilt-corrected held orientation (world<-body) for the matcher's absolute tilt reference.
	//! @p out_excess_m_s2 is |||f|| - g|; small means low linear acceleration.
	bool (*get_gravity_tilt_reference)(struct xrt_device *xdev,
	                                   struct xrt_quat *out_gravity_corrected_q,
	                                   double *out_excess_m_s2);
	//! Raw predicted (prior) pose for the matcher — the fusion's HONEST estimate, with NO body-lock / reach /
	//! re-entry reporting transforms, so the out-of-view visual ride never feeds back to mislead the matcher's
	//! gate + flip cost. @p out receives the pose (identity / untracked flags until the fusion is tracking).
	//! Optional — if NULL or it returns false, the caller falls back to the device's reported pose.
	bool (*get_predicted_pose)(struct xrt_device *xdev, timepoint_ns when_ns, struct xrt_space_relation *out);
	//! Milliseconds since the last position-constraining optical update at @p when_ns. Optional; if absent, the
	//! tracker treats predictive ROI as freshness-unknown and falls back to covariance-only gating.
	bool (*get_last_optical_age_ms)(struct xrt_device *xdev, timepoint_ns when_ns, double *age_ms);
	//! Predict one LED's image point @p out_zhat (px) and 2x2 innovation covariance @p out_S (row-major
	//! S00,S01,S10,S11 = H·P·Hᵀ + R) — the per-LED ANISOTROPIC gate ellipse from the fusion's live
	//! covariance, using the SAME projection/Jacobian as the per-LED fold (one source of truth). @p
	//! P_xrworld_cam + @p cam_calib + @p led_obj match @ref push_observed_leds exactly. Lets the
	//! front-end gate a candidate blob<->LED pairing by its Mahalanobis distance under the filter's
	//! real covariance (tight tilt / loose yaw after a dropout) instead of a fixed radius — the
	//! covariance-driven associator. Optional — may be NULL, or return false until the fusion is
	//! tracking (no usable prior, e.g. cold start); the caller then does NOT gate-fold.
	bool (*predict_led_gate)(struct xrt_device *xdev, timepoint_ns when_ns,
	                         const struct xrt_pose *P_xrworld_cam,
	                         const struct t_constellation_cam_calib *cam_calib,
	                         const struct xrt_vec3 *led_obj,
	                         float out_zhat[2], float out_S[4]);
	//! Position-only visual observation for frames where LED correspondences constrain translation but not
	//! orientation. refresh_optical_anchor is true only for independent visual positions that may reset optical
	//! freshness, body-lock, and optical velocity; weak single-view prior-supported positions must pass false.
	void (*push_observed_position)(struct xrt_device *xdev, timepoint_ns frame_mono_ns,
	                               const struct xrt_vec3 *position,
	                               const struct xrt_vec3 *position_variance,
	                               bool refresh_optical_anchor);
	//! Cache-only PnP pose candidate for per-LED divergence recovery. Optional; callers use it before
	//! ambiguous LED folds so the fusion can re-anchor to a guarded same-frame PnP solution if many LEDs
	//! are seen but the stale prior gates them out.
	void (*cache_pnp_pose_candidate)(struct xrt_device *xdev, timepoint_ns frame_mono_ns,
	                                 const struct xrt_pose *pose);
	//! World re-anchor (the head resnap guard's detector): the sampled HMD world pose stepped
	//! beyond its IMU envelope at @p frame_mono_ns — a SLAM relocalization/reset re-anchored the
	//! world by the rigid transform @p delta (x' = delta.q * x + delta.p, pivoted at the pre-step
	//! head position so head-relative geometry is preserved). The device transforms its
	//! world-frame fusion state by delta so its prior lands in the NEW world in the same camera
	//! frame the optical observations do. Optional — may be NULL.
	void (*notify_world_reanchor)(struct xrt_device *xdev, timepoint_ns frame_mono_ns,
	                             const struct xrt_pose *delta, const struct xrt_vec3 *new_raw_pivot,
	                             timepoint_ns publication_ns, double gyro_dps, double speed_mps);
};

struct t_constellation_tracked_device_connection *
t_constellation_tracker_add_device(struct t_constellation_tracker *ct,
                                   struct xrt_device *xdev,
                                   struct t_constellation_tracked_device_callbacks *cb);
void
t_constellation_tracked_device_connection_disconnect(struct t_constellation_tracked_device_connection *ctdc);

/*!
 * Debug/test only: count of frames fully processed through the pipeline (incremented exactly once per
 * frame at every pipeline exit). Lets an offline driver barrier on per-frame completion instead of
 * racing a fixed sleep. Thread-safe (locks the tracker's analysis lock).
 */
uint64_t
t_constellation_tracker_debug_frames_completed(struct t_constellation_tracker *ct);

#ifdef __cplusplus
}
#endif
