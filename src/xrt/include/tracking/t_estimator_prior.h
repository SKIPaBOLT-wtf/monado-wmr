// SPDX-License-Identifier: BSL-1.0
#pragma once
#include "xrt/xrt_tracking.h"
#include <stdbool.h>
#include <stdint.h>

//! Immutable estimation prior for one camera exposure, in one raw tracking-world generation.
//! Covariance is row-major [position xyz, GLOBAL orientation xyz], including cross terms.
//! A supported query may return valid=false: use cold search, never a rendered-pose fallback.
struct t_estimator_prior
{
	int64_t timestamp_ns;
	int64_t source_timestamp_ns;
	uint64_t world_generation;
	struct xrt_space_relation relation;
	double pose_covariance[36];
	double position_std_m, orientation_std_rad, yaw_std_rad, tilt_std_rad;
	struct xrt_quat gravity_orientation;
	double gravity_excess_m_s2;
	double optical_age_ms;
	bool valid;
	bool gravity_valid;
};
