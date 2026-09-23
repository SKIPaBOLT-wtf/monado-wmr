// Copyright 2021-2024, Collabora, Ltd.
// Copyright 2025, Beyley Cardellio
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief SLAM tracking code.
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @author Mateo de Mayo <mateo.demayo@collabora.com>
 * @author Beyley Cardellio <ep1cm1n10n123@gmail.com>
 * @ingroup aux_tracking
 */

#include "xrt/xrt_defines.h"

#include "util/u_logging.h"
#include "util/u_time.h"

#include "math/m_filter_fifo.h"
#include "math/m_vec3.h"
#include "math/m_predict.h"
#include "math/m_api.h"

#include "t_dead_reckoning.h"

#include <math.h>


//! An uncovered history interval is not a motion measurement. Preserve only the known base pose;
//! callers can keep it visible while reporting that current tracking/prediction is unavailable.
static void
uncovered_history_fallback(const struct xrt_space_relation *base, struct xrt_space_relation *out)
{
	*out = *base;
	out->relation_flags = (enum xrt_space_relation_flags)(base->relation_flags &
	    (XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_POSITION_VALID_BIT));
	if (!isfinite(out->pose.position.x) || !isfinite(out->pose.position.y) || !isfinite(out->pose.position.z)) {
		out->relation_flags = (enum xrt_space_relation_flags)(out->relation_flags & ~XRT_SPACE_RELATION_POSITION_VALID_BIT);
	}
	if (!isfinite(out->pose.orientation.x) || !isfinite(out->pose.orientation.y) ||
	    !isfinite(out->pose.orientation.z) || !isfinite(out->pose.orientation.w)) {
		out->relation_flags = (enum xrt_space_relation_flags)(out->relation_flags & ~XRT_SPACE_RELATION_ORIENTATION_VALID_BIT);
	}
	if (!(out->relation_flags & XRT_SPACE_RELATION_POSITION_VALID_BIT)) { out->pose.position = (struct xrt_vec3){0}; }
	if (!(out->relation_flags & XRT_SPACE_RELATION_ORIENTATION_VALID_BIT)) { out->pose.orientation = (struct xrt_quat){0,0,0,1}; }
	out->linear_velocity = out->angular_velocity = (struct xrt_vec3){0};
}


void
t_apply_dead_reckoning(struct m_ff_vec3_f32 *gyro_ff,
                       struct m_ff_vec3_f32 *accel_ff,
                       const struct xrt_vec3 *gravity_correction,
                       timepoint_ns when_ns,
                       const struct xrt_space_relation *base_rel,
                       timepoint_ns base_rel_ts,
                       struct xrt_space_relation *out_relation)
{
	t_apply_dead_reckoning_with_gyro_bias(gyro_ff, accel_ff, gravity_correction, NULL,
	                                    when_ns, base_rel, base_rel_ts, out_relation);
}

void
t_apply_dead_reckoning_with_gyro_bias(struct m_ff_vec3_f32 *gyro_ff,
                                    struct m_ff_vec3_f32 *accel_ff,
                                    const struct xrt_vec3 *gravity_correction,
                                    const struct xrt_vec3 *gyro_bias,
                                    timepoint_ns when_ns,
                                    const struct xrt_space_relation *base_rel,
                                    timepoint_ns base_rel_ts,
                                    struct xrt_space_relation *out_relation)
{
	bool using_accel = accel_ff != NULL;
	if (when_ns == base_rel_ts) { *out_relation = *base_rel; return; }
	if (when_ns < base_rel_ts || gyro_ff == NULL ||
	    (gyro_bias && (!isfinite(gyro_bias->x) || !isfinite(gyro_bias->y) || !isfinite(gyro_bias->z)))) {
		uncovered_history_fallback(base_rel, out_relation);
		return;
	}

	// Locate a REAL sample at/before the SLAM base. FIFO accessors expose allocated slots, so
	// unused zero timestamps are not evidence of coverage. Equality at the oldest sample is valid.
	int i = 0;
	uint64_t imu_ts = 0;
	bool covered = false;
	while (m_ff_vec3_f32_get_timestamp(gyro_ff, i, &imu_ts) && imu_ts > 0 && imu_ts <= INT64_MAX) {
		if ((int64_t)imu_ts <= base_rel_ts) { covered = true; break; }
		i++;
	}
	if (covered && using_accel) {
		uint64_t accel_ts = 0;
		covered = m_ff_vec3_f32_get_timestamp(accel_ff, i, &accel_ts) && accel_ts == imu_ts;
	}
	if (!covered) {
		uncovered_history_fallback(base_rel, out_relation);
		return;
	}
	if ((int64_t)imu_ts < base_rel_ts) { i--; }
	if (i < 0) {
		uncovered_history_fallback(base_rel, out_relation);
		return;
	}

	struct xrt_space_relation integ_rel = *base_rel;
	timepoint_ns integ_rel_ts = base_rel_ts;
	struct xrt_quat *orient = &integ_rel.pose.orientation;
	struct xrt_vec3 *pos = &integ_rel.pose.position;
	struct xrt_vec3 *ang_vel = &integ_rel.angular_velocity;
	struct xrt_vec3 *lin_vel = &integ_rel.linear_velocity;
	bool clamped = false; // If when_ns is older than the latest IMU ts

	while (i >= 0) { // Decreasing i increases timestamp
		// Get samples
		struct xrt_vec3 gyro = XRT_VEC3_ZERO;
		struct xrt_vec3 accel = XRT_VEC3_ZERO;
		uint64_t gyro_ts = 0;
		uint64_t accel_ts = 0;
		bool got = true;
		got &= m_ff_vec3_f32_get(gyro_ff, i, &gyro, &gyro_ts);
		if (using_accel) {
			got &= m_ff_vec3_f32_get(accel_ff, i, &accel, &accel_ts);
		}
		if (gyro_bias != NULL) {
			gyro = m_vec3_sub(gyro, *gyro_bias);
		}
		// A failed, unpaired, non-finite or backwards read invalidates the whole attempted
		// prediction, including any partially integrated prefix. Never integrate timestamp zero.
		if (!got || gyro_ts == 0 || gyro_ts > INT64_MAX ||
		    (using_accel && gyro_ts != accel_ts) || (int64_t)gyro_ts < integ_rel_ts ||
		    !isfinite(gyro.x) || !isfinite(gyro.y) || !isfinite(gyro.z) ||
		    (using_accel && (!isfinite(accel.x) || !isfinite(accel.y) || !isfinite(accel.z)))) {
			uncovered_history_fallback(base_rel, out_relation);
			return;
		}
		timepoint_ns ts = (timepoint_ns)gyro_ts;
		if (ts > when_ns) {
			clamped = true;
			// Preserve the existing right-endpoint residual integration model.
			ts = when_ns;
		}

		// Update time
		float dt = (float)time_ns_to_s(ts - integ_rel_ts);
		integ_rel_ts = ts;

		// Integrate gyroscope
		struct xrt_quat angvel_delta = XRT_QUAT_IDENTITY;
		struct xrt_vec3 scaled_half_g = m_vec3_mul_scalar(gyro, dt * 0.5f);
		math_quat_exp(&scaled_half_g, &angvel_delta);        // Same as using math_quat_from_angle_vector(g/dt)
		math_quat_rotate(orient, &angvel_delta, orient);     // Orientation
		math_quat_rotate_derivative(orient, &gyro, ang_vel); // Angular velocity
		if (integ_rel.relation_flags & XRT_SPACE_RELATION_ORIENTATION_VALID_BIT) {
			integ_rel.relation_flags = (enum xrt_space_relation_flags)(integ_rel.relation_flags |
			    XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT);
		}

		if (using_accel) {
			// Integrate accelerometer
			struct xrt_vec3 world_accel = XRT_VEC3_ZERO;
			math_quat_rotate_vec3(orient, &accel, &world_accel);
			world_accel = m_vec3_add(world_accel, *gravity_correction);
			// Integrate position with the velocity at the start of this interval.
			const struct xrt_vec3 accumulated_position_change = m_vec3_add(      //
			    m_vec3_mul_scalar(*lin_vel, dt),                                 //
			    m_vec3_mul_scalar(world_accel, dt * dt * 0.5f));                 //

			math_vec3_accum(&accumulated_position_change, pos);
			*lin_vel = m_vec3_add(*lin_vel, m_vec3_mul_scalar(world_accel, dt)); // Linear velocity
		}

		if (clamped) {
			break;
		}
		i--;
	}

	// Do the prediction based on the updated relation
	double last_imu_to_now_dt = time_ns_to_s(when_ns - integ_rel_ts);
	struct xrt_space_relation predicted_relation = XRT_SPACE_RELATION_ZERO;
	m_predict_relation(&integ_rel, last_imu_to_now_dt, &predicted_relation);

	*out_relation = predicted_relation;
}
