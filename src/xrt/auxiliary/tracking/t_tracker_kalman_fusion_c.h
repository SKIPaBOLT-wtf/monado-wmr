// Copyright 2024, Joel Valenciano
// Copyright 2025, Beyley Cardellio
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  C interface to generalized kalman filter.
 *
 * @author Joel Valenciano <joelv1907@gmail.com>
 * @author Beyley Cardellio <ep1cm1n10n123@gmail.com>
 * @ingroup aux_tracking
 */

#pragma once
#include "xrt/xrt_tracking.h"
#include "tracking/t_estimator_prior.h"

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct KalmanFusionInterfaceWrapper;

//! Diagnostic snapshot of the out-of-view report inputs at one timestamp. This is not used by production
//! tracking decisions; offline replay writes it to CSV so OOV policies can be evaluated against captured GT.
struct kalman_fusion_oov_debug
{
	bool valid;
	bool body_valid;
	double age_ms;
	double position_var_max;
	double inertial_var;
	double body_var;
	struct xrt_vec3 raw_predicted_position;
	struct xrt_vec3 optical_hold_position;
	struct xrt_vec3 body_report_position;
	struct xrt_vec3 raw_velocity;
	struct xrt_vec3 raw_acceleration;
	struct xrt_vec3 angular_velocity;
	double gravity_excess_m_s2;
};

//! One matched constellation LED for the tightly-coupled per-LED fusion update. C-accessible mirror
//! of the C++ LEDObservation (the C++ header aliases this exact struct, so there is one definition).
struct kalman_led_observation
{
	struct xrt_vec2 observed_px; //!< measured blob centroid (undistorted, normalized image coords)
	struct xrt_vec3 led_obj;     //!< LED position in the object frame (caller applies any frame flip)
	float pos_var_px2;          //!< optional isotropic centroid variance; <= 0 uses the fusion default
};

//! Pinhole camera + world->camera extrinsic for one constellation view. For normalized observations
//! pass fx=fy=1, cx=cy=0; the extrinsic must map the FILTER's world frame to the camera frame (the
//! caller folds the OpenXR<->OpenCV flip into this extrinsic and into led_obj).
struct kalman_led_camera_view
{
	float fx, fy, cx, cy;
	struct xrt_quat cam_world_orient; //!< world->camera rotation
	struct xrt_vec3 cam_world_pos;    //!< world->camera translation
};

//! Read-only historical estimator prior. Missing history is false with an initialized invalid bundle.
bool
kalman_fusion_get_estimator_prior(struct KalmanFusionInterfaceWrapper *wrapper, int64_t timestamp_ns,
                                 struct t_estimator_prior *out_prior);
uint64_t
kalman_fusion_get_world_generation(struct KalmanFusionInterfaceWrapper *wrapper);
//! Pure projection of a frozen prior, with the same Jacobian and pixel noise as the LED update.
bool
kalman_fusion_predict_led_gate_from_prior(const struct t_estimator_prior *prior,
                                         const struct kalman_led_observation *obs,
                                         const struct kalman_led_camera_view *view,
                                         float out_zhat[2], float out_S[4]);

struct KalmanFusionInterfaceWrapper *
kalman_fusion_create(void);

void
kalman_fusion_add_ui(struct KalmanFusionInterfaceWrapper *wrapper, void *root, const char *device_name);

void
kalman_fusion_destroy(struct KalmanFusionInterfaceWrapper *wrapper);

void
kalman_fusion_process_imu_data(struct KalmanFusionInterfaceWrapper *wrapper,
                               const struct xrt_imu_sample *sample,
                               const struct xrt_vec3 *accel_variance_optional,
                               const struct xrt_vec3 *gyro_variance_optional);

//! A hardware low-power status, not an IMU sample. Holds a recently confirmed rest state until real IMU resumes.
void
kalman_fusion_process_idle_status(struct KalmanFusionInterfaceWrapper *wrapper, timepoint_ns timestamp_ns);

//! @p hmd_world_pose is the LIVE HMD pose in the same world frame as the controller (NULL if
//! unavailable). It lets the fusion gate optical adoption on the controller-to-HMD distance (arm reach),
//! which is room-roam-invariant, and capture the body-lock offset for out-of-view reporting.
void
kalman_fusion_process_pose(struct KalmanFusionInterfaceWrapper *wrapper,
                           const struct xrt_pose_sample *sample,
                           const struct xrt_vec3 *position_variance_optional,
                           const struct xrt_vec3 *orientation_variance_optional,
                           float residual_limit,
                           const struct xrt_pose *hmd_world_pose);

//! Position-only optical observation for frames where visual evidence constrains translation but orientation is
//! ambiguous. Does not directly change orientation.
void
kalman_fusion_process_position(struct KalmanFusionInterfaceWrapper *wrapper,
                               timepoint_ns timestamp_ns,
                               const struct xrt_vec3 *position,
                               const struct xrt_vec3 *position_variance_optional,
                               const struct xrt_pose *hmd_world_pose,
                               bool refresh_optical_anchor);

//! Cache-only PnP candidate for per-LED divergence recovery. Updates only the guarded re-anchor target;
//! does not adopt the pose, change optical freshness, or report tracking.
void
kalman_fusion_cache_pnp_pose_candidate(struct KalmanFusionInterfaceWrapper *wrapper,
                                       timepoint_ns timestamp_ns,
                                       const struct xrt_pose *pose,
                                       const struct xrt_pose *hmd_world_pose);

//! Tightly-coupled per-LED optical update (the ESKF feed). @p feed=false runs a read-only diagnostic
//! (computes the per-LED reprojection RMS vs the current pose) instead of folding — used to verify
//! the frame/undistort transforms live before trusting the feed. feed=true returns the number of LEDs
//! folded this frame (>=0; -1 if the filter is not yet tracking); feed=false returns the diagnostic
//! reprojection RMS in NORMALIZED image units (or -1 if no pose).
//! @p hmd_world_pose: see kalman_fusion_process_pose (NULL if unavailable).
float
kalman_fusion_process_led_observations(struct KalmanFusionInterfaceWrapper *wrapper,
                                       timepoint_ns timestamp_ns,
                                       const struct kalman_led_observation *obs,
                                       size_t obs_count,
                                       const struct kalman_led_camera_view *view,
                                       const struct xrt_vec2 *pixel_variance_optional,
                                       float max_innov_px,
                                       bool feed,
                                       const struct xrt_pose *hmd_world_pose);

//! @p hmd_world_pose is the LIVE HMD pose (same world frame as the controller; NULL if unavailable).
//! When the controller's position is no longer optically observable, the fusion body-locks: it reports
//! the controller at its last controller-to-HMD offset riding with this live HMD pose (so an out-of-view
//! controller stays at arm's reach of the head instead of dead-reckoning to metres). NULL falls back to
//! the world-frame hold at the last optically-observed position.
void
kalman_fusion_get_prediction(struct KalmanFusionInterfaceWrapper *wrapper,
                             const timepoint_ns timestamp_ns,
                             struct xrt_space_relation *out_relation,
                             const struct xrt_pose *hmd_world_pose);

//! Raw predicted pose (the filter's honest belief, NO body-lock/reach/re-entry) for the constellation
//! matcher's PRIOR — distinct from kalman_fusion_get_prediction's compositor report, so the visual ride
//! never feeds back to mislead the front-end. Identity / untracked flags until the filter is tracking.
void
kalman_fusion_get_predicted_pose(struct KalmanFusionInterfaceWrapper *wrapper,
                                 const timepoint_ns timestamp_ns,
                                 struct xrt_space_relation *out_relation);

//! Predict one LED's image point @p out_zhat and 2x2 innovation covariance @p out_S (row-major:
//! S00,S01,S10,S11) = H P H^T + R, the per-LED gate ellipse for the covariance-driven associator. Uses
//! the SAME H/projection as the tightly-coupled fold (one source of truth). Returns false (writes
//! nothing) until the filter is tracking, or on a null wrapper/view.
bool
kalman_fusion_predict_led_gate(struct KalmanFusionInterfaceWrapper *wrapper,
                               const timepoint_ns timestamp_ns,
                               const struct kalman_led_observation *obs,
                               const struct kalman_led_camera_view *view,
                               float out_zhat[2],
                               float out_S[4]);

//! Current 1-sigma uncertainty of the predicted pose from the filter covariance: position_std (m) and
//! orientation_std (rad) are the worst-direction stds for sizing a prior-consistency gate; yaw_std (rad),
//! when non-null, is the orientation std about world-up ALONE — the uncertain yaw DoF (tilt is
//! gravity-anchored/observable) — for the mirror-flip cost's yaw scale; tilt_std (rad), when non-null, is
//! the worst-direction std within the horizontal (gravity-observable) plane — for the prior's tilt scale.
//! Any out may be null. Returns false until tracking.
bool
kalman_fusion_get_pose_uncertainty(struct KalmanFusionInterfaceWrapper *wrapper,
                                   double *position_std,
                                   double *orientation_std,
                                   double *yaw_std,
                                   double *tilt_std);

//! Gravity-tilt-corrected held orientation (world<-body), with yaw preserved and tilt snapped to
//! accelerometer gravity. @p out_excess_m_s2 is |||f|| - g|; small means low linear acceleration.
bool
kalman_fusion_get_gravity_tilt_reference(struct KalmanFusionInterfaceWrapper *wrapper,
                                         struct xrt_quat *out_gravity_corrected_q,
                                         double *out_excess_m_s2);

//! Cross-session IMU calibration cache (persisted per controller by the driver). set_* seeds a prior
//! before tracking; get_* reads the current converged gyro/accel bias + accel scale and returns true
//! only once a calibrated stance has occurred (estimate trustworthy to persist).
void
kalman_fusion_set_imu_calibration(struct KalmanFusionInterfaceWrapper *wrapper,
                                  const double gyro_bias[3],
                                  const double accel_bias[3],
                                  double accel_scale);
bool
kalman_fusion_get_imu_calibration(struct KalmanFusionInterfaceWrapper *wrapper,
                                  double gyro_bias[3],
                                  double accel_bias[3],
                                  double *accel_scale);

//! Optically-derived IMU intrinsics (offline-computed, persisted per serial alongside the bias cache).
//! @p gyro_correction (row-major 3x3) corrects the gyro: rate = M_g·(gyro - bg) — scale + gyro->device
//! misalignment in one matrix (the M imu_calib_from_optical.py fits). @p accel_correction (row-major 3x3)
//! is the accel ellipsoid T_a: specific force = T_a·(accel - ba). Either may be identity (uncorrected).
//! set_* applies a prior before tracking; get_* reads the applied intrinsics and returns false when
//! neither is a real correction (nothing to persist).
void
kalman_fusion_set_imu_intrinsics(struct KalmanFusionInterfaceWrapper *wrapper,
                                 const double gyro_correction[9],
                                 const double accel_correction[9]);
bool
kalman_fusion_get_imu_intrinsics(struct KalmanFusionInterfaceWrapper *wrapper,
                                 double gyro_correction[9],
                                 double accel_correction[9]);

//! Out-of-view body-plausibility update against the live head pose @p hmd_pose. Out of view the fusion softly
//! anchors the controller to its last body-relative (rigid HMD-relative) point so the dead-reckon drift stays
//! bounded. Call per controller IMU sample; a no-op while in view.
void
kalman_fusion_update_body_anchor(struct KalmanFusionInterfaceWrapper *wrapper, const struct xrt_pose *hmd_pose);

//! World re-anchor: the head tracker's world frame moved by the rigid transform @p delta
//! (x' = delta.q * x + delta.p) in one detected step (SLAM relocalization/reset). Transforms the
//! filter's world-frame state so the prior lives in the new world; body-frame state is untouched.
//! See KalmanFusionInterface::re_anchor_world.
// Presentation-only pair: hmd_world_pose is the presented IMU reference, expressed in this
// tracking origin. The relation remains raw; apply out_from_raw rigidly to pose and velocities.
void
kalman_fusion_get_prediction_with_presentation(struct KalmanFusionInterfaceWrapper *wrapper,
    int64_t timestamp_ns, struct xrt_space_relation *out_relation,
    const struct xrt_pose *hmd_world_pose, struct xrt_pose *out_from_raw, uint64_t *out_generation);

void
kalman_fusion_re_anchor_world_with_presentation(struct KalmanFusionInterfaceWrapper *wrapper,
    const struct xrt_pose *delta, const struct xrt_vec3 *new_raw_pivot,
    int64_t publication_ns, double gyro_dps, double speed_mps);

void
kalman_fusion_re_anchor_world(struct KalmanFusionInterfaceWrapper *wrapper, const struct xrt_pose *delta);

//! Diagnostics: named report regime from the last kalman_fusion_get_prediction call. Codes:
//! 0 Invalid, 1 VisualAccuracy, 2 InertialFastMotion, 3 WorldLocked, 4 BodyLocked, 5 ConfusedPosition.
int
kalman_fusion_debug_get_fusion_state(struct KalmanFusionInterfaceWrapper *wrapper,
                                     char *name_out,
                                     size_t name_cap);

//! Diagnostics: milliseconds since the last position-constraining optical update at @p timestamp_ns.
bool
kalman_fusion_debug_get_last_optical_age_ms(struct KalmanFusionInterfaceWrapper *wrapper,
                                            timepoint_ns timestamp_ns,
                                            double *age_ms);

//! Diagnostics: read raw, optical-hold, and body-report position candidates for OOV analysis.
bool
kalman_fusion_debug_get_oov_report(struct KalmanFusionInterfaceWrapper *wrapper,
                                   timepoint_ns timestamp_ns,
                                   const struct xrt_pose *hmd_world_pose,
                                   struct kalman_fusion_oov_debug *out_debug);

#ifdef __cplusplus
}
#endif // __cplusplus
