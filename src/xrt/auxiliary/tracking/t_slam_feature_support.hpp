// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
#pragma once

#include "vit/vit_interface.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

// A diagnostic of the 2D distribution returned with one immutable VIT pose.
// These are projected landmark coordinates, not inlier residuals or a pose confidence score.
struct t_slam_feature_support
{
	bool available = false;
	int64_t count = -1;
	int64_t finite_count = -1;
	double major_rms = -1;
	double minor_rms = -1;
};

static inline t_slam_feature_support
t_slam_summarize_feature_support(const vit_pose_features_t &features)
{
	t_slam_feature_support out{};
	if (features.count != 0 && features.features == nullptr) {
		return out;
	}
	out.available = true;
	out.count = features.count;
	out.finite_count = 0;
	double mean_u = 0, mean_v = 0, m2_uu = 0, m2_uv = 0, m2_vv = 0;
	for (uint32_t i = 0; i < features.count; ++i) {
		const auto &f = features.features[i];
		if (!std::isfinite(f.u) || !std::isfinite(f.v)) {
			continue;
		}
		const double n = ++out.finite_count;
		const double du = (double)f.u - mean_u;
		const double dv = (double)f.v - mean_v;
		mean_u += du / n;
		mean_v += dv / n;
		m2_uu += du * ((double)f.u - mean_u);
		m2_uv += du * ((double)f.v - mean_v);
		m2_vv += dv * ((double)f.v - mean_v);
	}
	if (out.finite_count == 0) {
		out.major_rms = out.minor_rms = 0;
		return out;
	}
	const double a = m2_uu / out.finite_count;
	const double b = m2_uv / out.finite_count;
	const double c = m2_vv / out.finite_count;
	const double half_trace = (a + c) / 2;
	const double radius = std::hypot((a - c) / 2, b);
	out.major_rms = std::sqrt(std::max(0.0, half_trace + radius));
	out.minor_rms = std::sqrt(std::max(0.0, half_trace - radius));
	return out;
}
