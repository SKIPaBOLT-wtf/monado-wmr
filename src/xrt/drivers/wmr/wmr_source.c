// Copyright 2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  WMR camera and IMU data source.
 * @author Mateo de Mayo <mateo.demayo@collabora.com>
 * @ingroup drv_wmr
 */
#include "wmr_source.h"
#include "wmr_camera.h"
#include "wmr_config.h"
#include "wmr_protocol.h"

#include "math/m_api.h"
#include "math/m_clock_tracking.h"
#include "math/m_filter_fifo.h"
#include "os/os_threading.h"
#include "util/u_debug.h"
#include "util/u_g2_telemetry.h"
#include "util/u_sink.h"
#include "util/u_var.h"
#include "util/u_trace_marker.h"
#include "xrt/xrt_tracking.h"
#include "xrt/xrt_frameserver.h"

#include <assert.h>
#include <stdio.h>

#define WMR_SOURCE_STR "WMR Source"

#define WMR_TRACE(w, ...) U_LOG_IFL_T(w->log_level, __VA_ARGS__)
#define WMR_DEBUG(w, ...) U_LOG_IFL_D(w->log_level, __VA_ARGS__)
#define WMR_INFO(w, ...) U_LOG_IFL_I(w->log_level, __VA_ARGS__)
#define WMR_WARN(w, ...) U_LOG_IFL_W(w->log_level, __VA_ARGS__)
#define WMR_ERROR(w, ...) U_LOG_IFL_E(w->log_level, __VA_ARGS__)
#define WMR_ASSERT(predicate, ...)                                                                                     \
	do {                                                                                                           \
		bool p = predicate;                                                                                    \
		if (!p) {                                                                                              \
			U_LOG(U_LOGGING_ERROR, __VA_ARGS__);                                                           \
			assert(false && "WMR_ASSERT failed: " #predicate);                                             \
			exit(EXIT_FAILURE);                                                                            \
		}                                                                                                      \
	} while (false);
#define WMR_ASSERT_(predicate) WMR_ASSERT(predicate, "Assertion failed " #predicate)

DEBUG_GET_ONCE_LOG_OPTION(wmr_log, "WMR_LOG", U_LOGGING_INFO)
// Keep the fork's previous estimator available for a controlled comparison.
DEBUG_GET_ONCE_BOOL_OPTION(wmr_clock_windowed, "WMR_CLOCK_WINDOWED", false)

/*!
 * Handles all the data sources from the WMR driver
 *
 * @todo Currently only properly handling tracking cameras, move IMU and other sources here
 * @implements xrt_fs
 * @implements xrt_frame_node
 */
struct wmr_source
{
	struct xrt_fs xfs;
	struct xrt_frame_node node;
	enum u_logging_level log_level; //!< Log level

	struct wmr_hmd_config config;
	struct wmr_camera *camera;

	// Sinks (head tracking)
	struct xrt_frame_sink cam_slam_sinks[WMR_MAX_CAMERAS]; //!< Intermediate sinks for camera frames
	struct xrt_imu_sink imu_sink;                          //!< Intermediate sink for IMU samples
	struct xrt_slam_sinks in_slam_sinks;                   //!< Pointers to intermediate sinks
	struct xrt_slam_sinks out_slam_sinks;                  //!< Pointers to downstream sinks

	struct xrt_frame_sink in_controller_sink;   //!< Sink that receives controller frames
	struct xrt_frame_sink *out_controller_sink; //!< Sink to send controller frames to tracker

	// UI Sinks (head tracking)
	struct u_sink_debug ui_slam_cam_sinks[WMR_MAX_CAMERAS]; //!< Sink to display camera frames in UI
	struct m_ff_vec3_f32 *gyro_ff;                          //!< Queue of gyroscope data to display in UI
	struct m_ff_vec3_f32 *accel_ff;                         //!< Queue of accelerometer data to display in UI

	bool is_running;          //!< Whether the device is streaming
	bool first_imu_received;  //!< Don't send frames until first IMU sample
	timepoint_ns last_imu_ns; //!< Last timepoint received.

	/*! One hw->mono authority for IMU and cameras, guarded by hw2mono_lock. Use the
	 * original arrival-offset filter by default; retain the windowed estimator for A/B. */
	struct os_mutex hw2mono_lock;
	struct m_clock_windowed_skew_tracker *hw2mono_clock;
	bool use_windowed_clock;
	bool hw2mono_valid;
	time_duration_ns hw2mono;
	timepoint_ns last_imu_hw_ns;
	timepoint_ns last_imu_arrival_ns;
	uint32_t imu_late_run;
	uint64_t imu_stabilised_total;

	/*! hw2mono offset sampled once per camera group (at cam0) so all views of a group and
	 * the interleaved controller frames convert identically. Only touched from the camera
	 * USB thread. */
	bool cam_hw2mono_valid;
	time_duration_ns cam_hw2mono;
};

/*! hw2mono window: 4 s of HMD IMU samples (1000 Hz; 4 samples per USB packet share one
 * arrival instant, and the min-skew logic keys on the newest, lowest-delay one). Sized from
 * a sweep over four live captures (results/hw2mono-windowed-20260708): 4 s brings the
 * steady-state per-camera-group offset step to ~0.1 us (~350x below the old exponential
 * filter) with zero converted-timeline regressions, bridges multi-second USB churn bursts,
 * and bounds the worst-case drift-following lag to ~0.2 ms (true drift ~56 us/s). */
#define WMR_HW2MONO_WINDOW_SAMPLES 4000

/*
 *
 * Sinks functionality
 *
 */

//! Sample the hw2mono authority for camera conversions (see cam_hw2mono).
static void
wmr_source_update_cam_hw2mono(struct wmr_source *ws, timepoint_ns group_hw_ts)
{
	os_mutex_lock(&ws->hw2mono_lock);
	timepoint_ns mono_ts = group_hw_ts + ws->hw2mono;
	bool have = ws->hw2mono_valid;
	if (ws->use_windowed_clock) {
		have = m_clock_windowed_skew_tracker_to_local(ws->hw2mono_clock, group_hw_ts, &mono_ts);
	}
	os_mutex_unlock(&ws->hw2mono_lock);
	if (have) {
		ws->cam_hw2mono = mono_ts - group_hw_ts;
		ws->cam_hw2mono_valid = true;
	}
}

#define DEFINE_RECEIVE_CAM(cam_id)                                                                                     \
	static void receive_cam##cam_id(struct xrt_frame_sink *sink, struct xrt_frame *xf)                             \
	{                                                                                                              \
		struct wmr_source *ws = container_of(sink, struct wmr_source, cam_slam_sinks[cam_id]);                 \
		if (cam_id == 0) {                                                                                     \
			wmr_source_update_cam_hw2mono(ws, (timepoint_ns)xf->timestamp);                                \
		}                                                                                                      \
		xf->timestamp += ws->cam_hw2mono;                                                                      \
		WMR_TRACE(ws, "cam" #cam_id " img t=%" PRId64 " source_t=%" PRId64, xf->timestamp,                     \
		          xf->source_timestamp);                                                                       \
		u_sink_debug_push_frame(&ws->ui_slam_cam_sinks[cam_id], xf);                                           \
		if (ws->out_slam_sinks.cams[cam_id] && ws->first_imu_received) {                                       \
			xrt_sink_push_frame(ws->out_slam_sinks.cams[cam_id], xf);                                      \
		}                                                                                                      \
	}

DEFINE_RECEIVE_CAM(0)
DEFINE_RECEIVE_CAM(1)
DEFINE_RECEIVE_CAM(2)
DEFINE_RECEIVE_CAM(3)

static void
receive_controller_frame(struct xrt_frame_sink *sink, struct xrt_frame *xf)
{
	struct wmr_source *ws = container_of(sink, struct wmr_source, in_controller_sink);

	// Convert from device TS to system TS with the same group-sampled offset the SLAM camera
	// groups use (one conversion timeline for every camera product of the shared USB stream).
	// Until the first IMU sample seeds the estimator there is no conversion authority at all;
	// forwarding would hand the tracker raw device-clock stamps (the 2026-07-09 phantom
	// head-resnap class), so drop the frame exactly like the SLAM path is gated.
	if (!ws->cam_hw2mono_valid) {
		wmr_source_update_cam_hw2mono(ws, (timepoint_ns)xf->timestamp);
		if (!ws->cam_hw2mono_valid) {
			WMR_DEBUG(ws, "Dropping controller frame until the clock estimator synchronises, hw ts %" PRIu64,
			          xf->timestamp);
			return;
		}
	}
	xf->timestamp += ws->cam_hw2mono;

	timepoint_ns now_mono = (timepoint_ns)os_monotonic_get_ns();
	WMR_TRACE(ws, "img seq %" PRIu64 " mono_t=%" PRIu64 " t=%" PRId64 " source_t=%" PRId64, xf->source_sequence,
	          now_mono, xf->timestamp, xf->source_timestamp);

	if (ws->out_controller_sink != NULL) {
		xrt_sink_push_frame(ws->out_controller_sink, xf);
	}
}

//! Define a function for each WMR_MAX_CAMERAS and reference it in this array
void (*receive_cam[WMR_MAX_CAMERAS])(struct xrt_frame_sink *, struct xrt_frame *) = {
    receive_cam0, //
    receive_cam1, //
    receive_cam2, //
    receive_cam3, //
};

//! Called only by the IMU thread with hw2mono_lock held.
static timepoint_ns
wmr_source_original_hw2mono(struct wmr_source *ws, timepoint_ns now_hw, timepoint_ns now_mono)
{
	// Preserve the original filter's coefficient for this comparison. The old driver also
	// used 250 Hz here when forwarding all four samples of a 1 kHz IMU packet.
	const float imu_freq = 250.f;
	const time_duration_ns late_threshold_ns = 50 * U_TIME_1MS_IN_NS;
	const uint32_t reseed_after_late_run = 25;
	time_duration_ns observed_offset = now_mono - now_hw;

	// Wintch/reverb-g2 patch 0093: after an arrival stall, queued samples drain faster than
	// device time. Do not learn the stall as a new clock offset. A sustained offset change
	// with a normally spaced arrival may re-seed; unlike the old filter this avoids dragging
	// the first live sample seconds into the future after a backlog.
	if (ws->hw2mono_valid && observed_offset - ws->hw2mono > late_threshold_ns) {
		time_duration_ns mono_gap = now_mono - ws->last_imu_arrival_ns;
		time_duration_ns hw_gap = now_hw - ws->last_imu_hw_ns;
		bool draining = ws->last_imu_arrival_ns != 0 && hw_gap > 0 && mono_gap * 2 < hw_gap;
		ws->imu_late_run++;
		if (!draining && ws->imu_late_run >= reseed_after_late_run) {
			WMR_WARN(ws, "IMU clock offset re-seeded after %u late samples (%.1f ms change)",
			         ws->imu_late_run, (double)(observed_offset - ws->hw2mono) / 1e6);
			ws->hw2mono = observed_offset;
			ws->imu_late_run = 0;
		} else if (ws->imu_late_run == 1) {
			WMR_INFO(ws, "Holding IMU clock offset during late arrivals (%.1f ms late)",
			         (double)(observed_offset - ws->hw2mono) / 1e6);
		}
	} else {
		if (ws->imu_late_run > 0) {
			WMR_INFO(ws, "IMU arrivals recovered after %u held-offset samples", ws->imu_late_run);
		}
		ws->imu_late_run = 0;
		m_clock_offset_a2b(imu_freq, now_hw, now_mono, &ws->hw2mono);
	}
	ws->hw2mono_valid = true;
	ws->last_imu_hw_ns = now_hw;
	ws->last_imu_arrival_ns = now_mono;
	return now_hw + ws->hw2mono;
}

static void
receive_imu_sample(struct xrt_imu_sink *sink, struct xrt_imu_sample *s)
{
	struct wmr_source *ws = container_of(sink, struct wmr_source, imu_sink);

	// Update the hw->mono offset estimate and convert the hardware timestamp into the
	// monotonic clock. Only IMU samples feed the estimator: they have the smallest USB
	// transmission time, and every other stream of the device shares its clock.
	timepoint_ns now_hw = s->timestamp_ns;
	timepoint_ns now_mono = (timepoint_ns)os_monotonic_get_ns();
	os_mutex_lock(&ws->hw2mono_lock);
	timepoint_ns ts = 0;
	bool have = true;
	if (ws->use_windowed_clock) {
		m_clock_windowed_skew_tracker_push(ws->hw2mono_clock, now_mono, now_hw);
		have = m_clock_windowed_skew_tracker_to_local(ws->hw2mono_clock, now_hw, &ts);
	} else {
		ts = wmr_source_original_hw2mono(ws, now_hw, now_mono);
	}
	os_mutex_unlock(&ws->hw2mono_lock);
	if (!have) {
		WMR_DEBUG(ws, "Dropping IMU sample until the clock estimator synchronises, hw ts %" PRId64, now_hw);
		return;
	}

	// Wintch/reverb-g2 patch 0022: preserve a sample when the arrival-offset filter causes
	// a small backwards step. Keep this floor local to the IMU sample: feeding it back into
	// hw2mono would ratchet the shared camera/IMU clock forward. Large discontinuities still
	// take the existing rejection path, and the windowed A/B path remains unchanged.
	if (!ws->use_windowed_clock && ws->last_imu_ns >= ts &&
	    ws->last_imu_ns - ts < 20 * U_TIME_1MS_IN_NS) {
		ts = ws->last_imu_ns + 250 * U_TIME_1US_IN_NS;
		ws->imu_stabilised_total++;
		if (ws->imu_stabilised_total % 1000 == 1) {
			WMR_INFO(ws, "Preserved IMU sample after clock-offset jitter (%" PRIu64 " total)",
			         ws->imu_stabilised_total);
		}
	}

	/*
	 * Check if the timepoint does time travel, we get one or two
	 * old samples when the device has not been cleanly shut down.
	 */
	if (ws->last_imu_ns >= ts) { // >= : also drop a duplicate timestamp (dt==0) before it reaches SLAM
		WMR_WARN(ws, "Received sample from the past, new: %" PRId64 ", last: %" PRId64 ", diff: %" PRId64, ts,
		         ws->last_imu_ns, ws->last_imu_ns - ts);
		return;
	}

	ws->first_imu_received = true;
	ws->last_imu_ns = ts;
	s->timestamp_ns = ts;

	struct xrt_vec3_f64 a = s->accel_m_s2;
	struct xrt_vec3_f64 w = s->gyro_rad_secs;
	WMR_TRACE(ws, "imu t=%" PRId64 " a=(%f %f %f) w=(%f %f %f)", ts, a.x, a.y, a.z, w.x, w.y, w.z);

	// Push to debug UI
	struct xrt_vec3 gyro = {(float)w.x, (float)w.y, (float)w.z};
	struct xrt_vec3 accel = {(float)a.x, (float)a.y, (float)a.z};
	m_ff_vec3_f32_push(ws->gyro_ff, &gyro, ts);
	m_ff_vec3_f32_push(ws->accel_ff, &accel, ts);

	// Telemetry: HMD IMU sample (device_id 0). hw_ts_ns is the monotonic-converted
	// sensor ts (ts), in the common clock shared by every stream (MAJOR-3).
	g2_telem_imu(0, (uint64_t)ts, accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z);

	if (ws->out_slam_sinks.imu) {
		xrt_sink_push_imu(ws->out_slam_sinks.imu, s);
	}
}


/*
 *
 * Frameserver functionality
 *
 */

static inline struct wmr_source *
wmr_source_from_xfs(struct xrt_fs *xfs)
{
	struct wmr_source *ws = container_of(xfs, struct wmr_source, xfs);
	return ws;
}

static bool
wmr_source_enumerate_modes(struct xrt_fs *xfs, struct xrt_fs_mode **out_modes, uint32_t *out_count)
{
	WMR_ASSERT(false, "Not implemented");
	return false;
}

static bool
wmr_source_configure_capture(struct xrt_fs *xfs, struct xrt_fs_capture_parameters *cp)
{
	WMR_ASSERT(false, "Not implemented");
	return false;
}

static bool
wmr_source_stream_stop(struct xrt_fs *xfs)
{
	DRV_TRACE_MARKER();

	struct wmr_source *ws = wmr_source_from_xfs(xfs);

	bool stopped = wmr_camera_stop(ws->camera);
	if (!stopped) {
		WMR_ERROR(ws, "Unable to stop WMR cameras");
		WMR_ASSERT_(false);
	}

	return stopped;
}

static bool
wmr_source_is_running(struct xrt_fs *xfs)
{
	DRV_TRACE_MARKER();

	struct wmr_source *ws = wmr_source_from_xfs(xfs);
	return ws->is_running;
}

static bool
wmr_source_stream_start(struct xrt_fs *xfs,
                        struct xrt_frame_sink *xs,
                        enum xrt_fs_capture_type capture_type,
                        uint32_t descriptor_index)
{
	DRV_TRACE_MARKER();

	struct wmr_source *ws = wmr_source_from_xfs(xfs);

	if (xs == NULL && capture_type == XRT_FS_CAPTURE_TYPE_TRACKING) {
		WMR_INFO(ws, "Starting WMR stream in tracking mode");
	} else if (xs != NULL && capture_type == XRT_FS_CAPTURE_TYPE_CALIBRATION) {
		WMR_INFO(ws, "Starting WMR stream in calibration mode, will stream only cam0 frames");
		ws->out_slam_sinks.cam_count = 1;
		ws->out_slam_sinks.cams[0] = xs;
	} else {
		WMR_ASSERT(false, "Unsupported stream configuration xs=%p capture_type=%d", (void *)xs, capture_type);
		return false;
	}

	bool started = wmr_camera_start(ws->camera);
	if (!started) {
		WMR_ERROR(ws, "Unable to start WMR cameras");
		WMR_ASSERT_(false);
	}

	ws->is_running = started;
	return ws->is_running;
}

static bool
wmr_source_slam_stream_start(struct xrt_fs *xfs, struct xrt_slam_sinks *sinks)
{
	DRV_TRACE_MARKER();

	struct wmr_source *ws = wmr_source_from_xfs(xfs);
	if (sinks != NULL) {
		ws->out_slam_sinks = *sinks;
	}
	return wmr_source_stream_start(xfs, NULL, XRT_FS_CAPTURE_TYPE_TRACKING, 0);
}


/*
 *
 * Frame node functionality
 *
 */

static void
wmr_source_node_break_apart(struct xrt_frame_node *node)
{
	DRV_TRACE_MARKER();

	struct wmr_source *ws = container_of(node, struct wmr_source, node);
	wmr_source_stream_stop(&ws->xfs);
}

static void
wmr_source_node_destroy(struct xrt_frame_node *node)
{
	DRV_TRACE_MARKER();

	struct wmr_source *ws = container_of(node, struct wmr_source, node);
	WMR_DEBUG(ws, "Destroying WMR source");
	for (int i = 0; i < ws->config.tcam_count; i++) {
		u_sink_debug_destroy(&ws->ui_slam_cam_sinks[i]);
	}
	m_ff_vec3_f32_free(&ws->gyro_ff);
	m_ff_vec3_f32_free(&ws->accel_ff);
	m_clock_windowed_skew_tracker_destroy(ws->hw2mono_clock);
	os_mutex_destroy(&ws->hw2mono_lock);
	u_var_remove_root(ws);
	if (ws->camera != NULL) { // It could be null if XRT_HAVE_LIBUSB is not defined
		wmr_camera_free(ws->camera);
	}
	free(ws);
}


/*
 *
 * Exported functions
 *
 */

//! Create and open the frame server for IMU/camera streaming.
struct xrt_fs *
wmr_source_create(struct xrt_frame_context *xfctx,
                  struct xrt_prober_device *dev_holo,
                  struct wmr_hmd_config cfg,
                  struct xrt_frame_sink *out_controller_sink)
{
	DRV_TRACE_MARKER();

	struct wmr_source *ws = U_TYPED_CALLOC(struct wmr_source);
	ws->log_level = debug_get_log_option_wmr_log();

	ws->hw2mono_clock = m_clock_windowed_skew_tracker_alloc(WMR_HW2MONO_WINDOW_SAMPLES);
	os_mutex_init(&ws->hw2mono_lock);
	ws->use_windowed_clock = debug_get_bool_option_wmr_clock_windowed();
	WMR_INFO(ws, "WMR clock mode: %s", ws->use_windowed_clock ? "windowed" : "original with backlog guard");

	// Setup xrt_fs
	struct xrt_fs *xfs = &ws->xfs;
	xfs->enumerate_modes = wmr_source_enumerate_modes;
	xfs->configure_capture = wmr_source_configure_capture;
	xfs->stream_start = wmr_source_stream_start;
	xfs->slam_stream_start = wmr_source_slam_stream_start;
	xfs->stream_stop = wmr_source_stream_stop;
	xfs->is_running = wmr_source_is_running;
	(void)snprintf(xfs->name, sizeof(xfs->name), WMR_SOURCE_STR);
	(void)snprintf(xfs->product, sizeof(xfs->product), WMR_SOURCE_STR " Product");
	(void)snprintf(xfs->manufacturer, sizeof(xfs->manufacturer), WMR_SOURCE_STR " Manufacturer");
	(void)snprintf(xfs->serial, sizeof(xfs->serial), WMR_SOURCE_STR " Serial");
	xfs->source_id = *((uint64_t *)"WMR_SRC\0");

	// Setup sinks
	for (int i = 0; i < WMR_MAX_CAMERAS; i++) {
		ws->cam_slam_sinks[i].push_frame = receive_cam[i];
	}
	ws->imu_sink.push_imu = receive_imu_sample;

	ws->in_slam_sinks.cam_count = cfg.tcam_count;
	for (int i = 0; i < cfg.tcam_count; i++) {
		ws->in_slam_sinks.cams[i] = &ws->cam_slam_sinks[i];
	}
	ws->in_slam_sinks.imu = &ws->imu_sink;

	ws->in_controller_sink.push_frame = receive_controller_frame;
	ws->out_controller_sink = out_controller_sink;

	struct wmr_camera_open_config options = {
	    .dev_holo = dev_holo,
	    .tcam_confs = cfg.tcams,
	    .tcam_sinks = ws->in_slam_sinks.cams,
	    .tcam_count = cfg.tcam_count,
	    .slam_cam_count = cfg.slam_cam_count,
	    .controller_cam_sink = &ws->in_controller_sink,
	    .log_level = ws->log_level,
	};

	ws->camera = wmr_camera_open(&options);
	ws->config = cfg;

	// Setup UI
	for (int i = 0; i < cfg.tcam_count; i++) {
		u_sink_debug_init(&ws->ui_slam_cam_sinks[i]);
	}
	m_ff_vec3_f32_alloc(&ws->gyro_ff, 1000);
	m_ff_vec3_f32_alloc(&ws->accel_ff, 1000);
	u_var_add_root(ws, WMR_SOURCE_STR, false);
	u_var_add_log_level(ws, &ws->log_level, "Log Level");
	u_var_add_ro_ff_vec3_f32(ws, ws->gyro_ff, "Gyroscope");
	u_var_add_ro_ff_vec3_f32(ws, ws->accel_ff, "Accelerometer");
	for (int i = 0; i < cfg.tcam_count; i++) {
		char label[] = "Camera NNNNNNNNNNN SLAM";
		(void)snprintf(label, sizeof(label), "Camera %d SLAM", i);
		u_var_add_sink_debug(ws, &ws->ui_slam_cam_sinks[i], label);
	}

	// Setup node
	struct xrt_frame_node *xfn = &ws->node;
	xfn->break_apart = wmr_source_node_break_apart;
	xfn->destroy = wmr_source_node_destroy;
	xrt_frame_context_add(xfctx, &ws->node);

	WMR_DEBUG(ws, "WMR Source created");

	return xfs;
}

void
wmr_source_push_imu_packet(struct xrt_fs *xfs, timepoint_ns t, struct xrt_vec3 accel, struct xrt_vec3 gyro)
{
	DRV_TRACE_MARKER();
	struct wmr_source *ws = wmr_source_from_xfs(xfs);
	struct xrt_vec3_f64 accel_f64 = {accel.x, accel.y, accel.z};
	struct xrt_vec3_f64 gyro_f64 = {gyro.x, gyro.y, gyro.z};
	struct xrt_imu_sample sample = {.timestamp_ns = t, .accel_m_s2 = accel_f64, .gyro_rad_secs = gyro_f64};
	xrt_sink_push_imu(&ws->imu_sink, &sample);
}
