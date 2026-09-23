// SPDX-License-Identifier: BSL-1.0
// Historical negative controls: old unbounded mask freshness and notify-only cache.
// Preserved intentionally to demonstrate that regression tests catch both defects.
// This fixture is never built into the runtime.
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

			bool have_rect = false;
			struct pose_rect rect = {0, 0, 0, 0};
			sigma_px[i][d] = -1.0f;

			/* Prediction rect, inflated by the projected position uncertainty. Trusted
			 * exactly as far as the fusion can bound it: a coasting controller gets a
			 * honestly-larger rect, a lost/divergent one inflates past the area cap below
			 * and self-disables. */
			if (dev_state->prior_pos_std_m >= 0.0f) {
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

			/* Last-seen rect: world-anchored optical truth, re-projected through the live
			 * head pose — a lost-but-resting controller keeps emitting light there (the
			 * static-map exemption pattern). */
			if (dev_state->have_last_seen_pose &&
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
// One publication clock and pivot
for (int i = 0; i < ct->num_devices; i++) {
					struct t_constellation_tracked_device_connection *conn =
					    ct->devices[i].connection;
					if (conn != NULL) {
						constellation_tracked_device_connection_notify_world_reanchor(
						    conn, (timepoint_ns)xf->timestamp, &delta, &transformed_pivot.position,
						    publication_ns, gyro_dps, speed_mps);
					}
				}
