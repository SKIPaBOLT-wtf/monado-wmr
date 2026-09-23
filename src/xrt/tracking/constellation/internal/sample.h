// Copyright 2023, Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Constellation tracking details for 1 exposure sample
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */

#pragma once

#include "tracking/t_tracking.h"
#include "tracking/t_estimator_prior.h"

#include "xrt/xrt_defines.h"
#include "xrt/xrt_frame.h"

#include "blobwatch.h"
#include "pose_metrics.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CONSTELLATION_MAX_DEVICES 4
#define CONSTELLATION_MAX_CAMERAS XRT_TRACKING_MAX_SLAM_CAMS

/* Information about one device being tracked in this sample */
struct tracking_sample_device_state
{
	/* Index into the devices array for this state info */
	int dev_index;
	const struct t_estimator_prior *estimator_prior; //!< points into owning sample, NULL for legacy devices

	struct t_constellation_led_model *led_model;

	/* Predicted device pose and error bounds from fusion, in world space */
	struct xrt_pose P_world_obj_prior;
	struct xrt_vec3 prior_pos_error;
	struct xrt_vec3 prior_rot_error;
	/* Raw fusion position 1-sigma (m) behind prior_pos_error's clamped bound; <0 when the
	 * fusion exposes no uncertainty (not tracking). The SLAM controller-mask push inflates
	 * its prediction rect by this projected to pixels, so an untrustworthy prediction grows
	 * past the area cap and self-disables instead of masking a wrong region. */
	float prior_pos_std_m;
	/* Prior-orientation trust for the soft mirror-flip cost: true whenever the fusion is tracking, so the
	 * DRIFTLESS gravity-anchored prior TILT is a valid reference even through an optical dropout (the
	 * gyro-blind snap-back case). False at cold start, where the search runs on reprojection alone. */
	bool prior_tilt_trusted;
	bool prior_optical_stale;
	bool prior_position_tracked;
	bool prior_orientation_tracked;
	/* Live fusion yaw (orientation) 1-sigma uncertainty, in radians, for the soft mirror-flip cost's
	 * anisotropic yaw scale. Tight when the yaw prior is fresh (the prior term picks the prior-consistent
	 * twin); large after a long dropout / cold start (the yaw term vanishes, reprojection decides). */
	float prior_yaw_sigma_rad;
	/* Live fusion horizontal-plane (gravity-observable tilt) 1-sigma, in radians, for the prior cost's
	 * tilt scale — clamped to [FLIP_COST_TILT_SIGMA_MIN, GRAVITY_TILT_TOL]: tight when gravity-observed
	 * (a tilt twin is decisively penalised), honestly widened during violent dynamics. */
	float prior_tilt_sigma_rad;

	bool gravity_ref_valid;
	bool gravity_ref_clean;
	struct xrt_pose P_world_obj_gravity;

	/* Last observed pose, in world space */
	bool have_last_seen_pose;
	struct xrt_pose last_seen_pose;

	bool found_device_pose;     /* Set to true when the device was found in this sample */
	int found_pose_view_id;     /* Set to the camera ID where the device was found */
	struct xrt_pose final_pose; /* Global pose that was detected */

	struct pose_metrics score;
	struct pose_metrics_blob_match_info blob_match_info;

	/* Bitmask of views whose matched LEDs have already been folded into the fusion this sample, so
	 * the accepted + recover + prior-refine paths cannot fold the same view's LEDs twice. */
	uint16_t led_emit_view_mask;
};

/* Information about 1 camera frame in this sample */
struct tracking_sample_frame
{
	/* Video frame data we are analysing */
	struct xrt_frame *vframe;

	/* The pose from which this view is observed (cam wrt world) */
	struct xrt_pose P_world_cam;
	/* Inverse of the pose from which this view is observed (world wrt camera) */
	struct xrt_pose P_cam_world;
	/* Gravity vector as observed from this camera */
	struct xrt_vec3 cam_gravity_vector;

	/* blobs observation and the owning blobwatch */
	blobwatch *bw;
	blobservation *bwobs;
};

struct constellation_tracking_sample
{
	uint64_t timestamp; // Exposure timestamp
	struct t_estimator_prior estimator_priors[CONSTELLATION_MAX_DEVICES];
	bool estimator_prior_supported[CONSTELLATION_MAX_DEVICES];

	/* Device poses at capture time */
	struct tracking_sample_device_state devices[CONSTELLATION_MAX_DEVICES];
	uint8_t n_devices;

	struct tracking_sample_frame views[CONSTELLATION_MAX_CAMERAS];
	uint8_t n_views;
};

struct constellation_tracking_sample *
constellation_tracking_sample_new(void);
void
constellation_tracking_sample_free(struct constellation_tracking_sample *sample);

#ifdef __cplusplus
}
#endif
