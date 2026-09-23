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

#include "t_tracker_kalman_fusion.hpp"
#include "xrt/xrt_tracking.h"

#include <cstdlib>
#include <vector>

using xrt::auxiliary::tracking::KalmanFusionInterface;

struct KalmanFusionInterfaceWrapper
{
	std::unique_ptr<KalmanFusionInterface> fusion;

	KalmanFusionInterfaceWrapper() : fusion(KalmanFusionInterface::create()) {}

	~KalmanFusionInterfaceWrapper() {}
};

extern "C" {

struct KalmanFusionInterfaceWrapper *
kalman_fusion_create(void)
{
	return new KalmanFusionInterfaceWrapper;
}

void
kalman_fusion_destroy(KalmanFusionInterfaceWrapper *wrapper)
{
	delete wrapper;
}

void
kalman_fusion_add_ui(struct KalmanFusionInterfaceWrapper *wrapper, void *root, const char *device_name)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->add_ui(root, device_name);
}

void
kalman_fusion_process_imu_data(KalmanFusionInterfaceWrapper *wrapper,
                               const struct xrt_imu_sample *sample,
                               const struct xrt_vec3 *accel_variance_optional,
                               const struct xrt_vec3 *gyro_variance_optional)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->process_imu_data(sample, accel_variance_optional, gyro_variance_optional);
}

void
kalman_fusion_process_idle_status(KalmanFusionInterfaceWrapper *wrapper, timepoint_ns timestamp_ns)
{
	if (wrapper != nullptr) {
		wrapper->fusion->process_idle_status(timestamp_ns);
	}
}

void
kalman_fusion_process_pose(KalmanFusionInterfaceWrapper *wrapper,
                           const struct xrt_pose_sample *sample,
                           const struct xrt_vec3 *position_variance_optional,
                           const struct xrt_vec3 *orientation_variance_optional,
                           const float residual_limit,
                           const struct xrt_pose *hmd_world_pose)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->process_pose(sample, position_variance_optional, orientation_variance_optional,
	                              residual_limit, hmd_world_pose);
}

float
kalman_fusion_process_led_observations(struct KalmanFusionInterfaceWrapper *wrapper,
                                       timepoint_ns timestamp_ns,
                                       const struct kalman_led_observation *obs,
                                       size_t obs_count,
                                       const struct kalman_led_camera_view *view,
                                       const struct xrt_vec2 *pixel_variance_optional,
                                       float max_innov_px,
                                       bool feed,
                                       const struct xrt_pose *hmd_world_pose)
{
	if (wrapper == nullptr || view == nullptr || (obs == nullptr && obs_count > 0)) {
		return -1.0f;
	}
	// LEDObservation is an alias of kalman_led_observation, so the C array constructs the vector
	// directly (no per-element conversion).
	std::vector<kalman_led_observation> v(obs, obs + obs_count);
	return wrapper->fusion->process_led_observations(timestamp_ns, v, *view, pixel_variance_optional,
	                                                 max_innov_px, feed, hmd_world_pose);
}

void
kalman_fusion_get_prediction(struct KalmanFusionInterfaceWrapper *wrapper,
                             const timepoint_ns timestamp_ns,
                             struct xrt_space_relation *out_relation,
                             const struct xrt_pose *hmd_world_pose)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->get_prediction(timestamp_ns, out_relation, hmd_world_pose);
}

void
kalman_fusion_get_predicted_pose(struct KalmanFusionInterfaceWrapper *wrapper,
                                 const timepoint_ns timestamp_ns,
                                 struct xrt_space_relation *out_relation)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->get_predicted_pose(timestamp_ns, out_relation);
}

void
kalman_fusion_process_position(struct KalmanFusionInterfaceWrapper *wrapper,
                               timepoint_ns timestamp_ns,
                               const struct xrt_vec3 *position,
                               const struct xrt_vec3 *position_variance_optional,
                               const struct xrt_pose *hmd_world_pose,
                               bool refresh_optical_anchor)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->process_position(timestamp_ns, position, position_variance_optional, hmd_world_pose,
	                                  refresh_optical_anchor);
}

void
kalman_fusion_cache_pnp_pose_candidate(struct KalmanFusionInterfaceWrapper *wrapper,
                                       timepoint_ns timestamp_ns,
                                       const struct xrt_pose *pose,
                                       const struct xrt_pose *hmd_world_pose)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->cache_pnp_pose_candidate(timestamp_ns, pose, hmd_world_pose);
}

bool
kalman_fusion_predict_led_gate(struct KalmanFusionInterfaceWrapper *wrapper,
                               const timepoint_ns timestamp_ns,
                               const struct kalman_led_observation *obs,
                               const struct kalman_led_camera_view *view,
                               float out_zhat[2],
                               float out_S[4])
{
	if (wrapper == nullptr || obs == nullptr || view == nullptr) {
		return false;
	}
	return wrapper->fusion->predict_led_gate(timestamp_ns, *obs, *view, out_zhat, out_S);
}

bool
kalman_fusion_get_pose_uncertainty(struct KalmanFusionInterfaceWrapper *wrapper,
                                   double *position_std,
                                   double *orientation_std,
                                   double *yaw_std,
                                   double *tilt_std)
{
	if (wrapper == nullptr) {
		return false;
	}
	return wrapper->fusion->get_pose_uncertainty(position_std, orientation_std, yaw_std, tilt_std);
}

bool
kalman_fusion_get_gravity_tilt_reference(struct KalmanFusionInterfaceWrapper *wrapper,
                                         struct xrt_quat *out_gravity_corrected_q,
                                         double *out_excess_m_s2)
{
	if (wrapper == nullptr) {
		return false;
	}
	return wrapper->fusion->get_gravity_tilt_reference(out_gravity_corrected_q, out_excess_m_s2);
}

void
kalman_fusion_set_imu_calibration(struct KalmanFusionInterfaceWrapper *wrapper,
                                  const double gyro_bias[3],
                                  const double accel_bias[3],
                                  double accel_scale)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->set_imu_calibration(gyro_bias, accel_bias, accel_scale);
}

bool
kalman_fusion_get_imu_calibration(struct KalmanFusionInterfaceWrapper *wrapper,
                                  double gyro_bias[3],
                                  double accel_bias[3],
                                  double *accel_scale)
{
	if (wrapper == nullptr) {
		return false;
	}
	return wrapper->fusion->get_imu_calibration(gyro_bias, accel_bias, accel_scale);
}

void
kalman_fusion_set_imu_intrinsics(struct KalmanFusionInterfaceWrapper *wrapper,
                                 const double gyro_correction[9],
                                 const double accel_correction[9])
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->set_imu_intrinsics(gyro_correction, accel_correction);
}

bool
kalman_fusion_get_imu_intrinsics(struct KalmanFusionInterfaceWrapper *wrapper,
                                 double gyro_correction[9],
                                 double accel_correction[9])
{
	if (wrapper == nullptr) {
		return false;
	}
	return wrapper->fusion->get_imu_intrinsics(gyro_correction, accel_correction);
}

void
kalman_fusion_update_body_anchor(struct KalmanFusionInterfaceWrapper *wrapper, const struct xrt_pose *hmd_pose)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->update_body_anchor(hmd_pose);
}

void
kalman_fusion_re_anchor_world(struct KalmanFusionInterfaceWrapper *wrapper, const struct xrt_pose *delta)
{
	if (wrapper == nullptr) {
		return;
	}
	wrapper->fusion->re_anchor_world(delta);
}

int
kalman_fusion_debug_get_fusion_state(struct KalmanFusionInterfaceWrapper *wrapper, char *name_out, size_t name_cap)
{
	if (wrapper == nullptr) {
		if (name_out != nullptr && name_cap > 0) {
			name_out[0] = '\0';
		}
		return 0;
	}
	return wrapper->fusion->debug_get_fusion_state(name_out, name_cap);
}

bool
kalman_fusion_debug_get_last_optical_age_ms(struct KalmanFusionInterfaceWrapper *wrapper,
                                            timepoint_ns timestamp_ns,
                                            double *age_ms)
{
	if (wrapper == nullptr) {
		return false;
	}
	return wrapper->fusion->debug_get_last_optical_age_ms(timestamp_ns, age_ms);
}

bool
kalman_fusion_debug_get_oov_report(struct KalmanFusionInterfaceWrapper *wrapper,
                                   timepoint_ns timestamp_ns,
                                   const struct xrt_pose *hmd_world_pose,
                                   struct kalman_fusion_oov_debug *out_debug)
{
	if (wrapper == nullptr) {
		return false;
	}
	return wrapper->fusion->debug_get_oov_report(timestamp_ns, hmd_world_pose, out_debug);
}
}

extern "C" void
kalman_fusion_get_prediction_with_presentation(struct KalmanFusionInterfaceWrapper *wrapper,
    int64_t timestamp_ns, struct xrt_space_relation *out_relation,
    const struct xrt_pose *hmd_world_pose, struct xrt_pose *out_from_raw, uint64_t *out_generation)
{
    if (!wrapper) return;
    wrapper->fusion->get_prediction(timestamp_ns, out_relation, hmd_world_pose, out_from_raw, out_generation, true);
}

extern "C" void
kalman_fusion_re_anchor_world_with_presentation(struct KalmanFusionInterfaceWrapper *wrapper,
    const struct xrt_pose *delta, const struct xrt_vec3 *new_raw_pivot,
    int64_t publication_ns, double gyro_dps, double speed_mps)
{
    if (!wrapper) return;
    wrapper->fusion->re_anchor_world_with_presentation(delta, new_raw_pivot, publication_ns, gyro_dps, speed_mps);
}


bool
kalman_fusion_get_estimator_prior(struct KalmanFusionInterfaceWrapper *wrapper, int64_t timestamp_ns,
                                 struct t_estimator_prior *out_prior)
{
	return wrapper != nullptr && wrapper->fusion->get_estimator_prior(timestamp_ns, out_prior);
}

uint64_t
kalman_fusion_get_world_generation(struct KalmanFusionInterfaceWrapper *wrapper)
{
	return wrapper != nullptr ? wrapper->fusion->get_world_generation() : UINT64_MAX;
}

bool
kalman_fusion_predict_led_gate_from_prior(const struct t_estimator_prior *prior,
                                         const struct kalman_led_observation *obs,
                                         const struct kalman_led_camera_view *view,
                                         float out_zhat[2], float out_S[4])
{
	return prior != nullptr && obs != nullptr && view != nullptr &&
	    xrt::auxiliary::tracking::predict_led_gate_from_prior(*prior, *obs, *view, out_zhat, out_S);
}
