// Copyright 2020-2021, N Madsen.
// Copyright 2020-2021, Collabora, Ltd.
// Copyright 2021-2023, Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
//
/*!
 * @file
 * @brief Common implementation for WMR controllers, handling
 * shared behaviour such as communication, configuration reading,
 * IMU integration.
 * @author Jan Schmidt <jan@centricular.com>
 * @author Nis Madsen <nima_zero_one@protonmail.com>
 * @ingroup drv_wmr
 */
#pragma once

#include "os/os_threading.h"
#include "math/m_clock_tracking.h"
#include "util/u_device.h"
#include "util/u_logging.h"
#include "util/u_var.h"
#include "xrt/xrt_device.h"
#include "tracking/t_led_models.h"
#include "tracking/t_constellation_tracking.h"
#include "tracking/t_tracker_kalman_fusion_c.h"

#include "wmr_common.h"
#include "wmr_controller_protocol.h"
#include "wmr_config.h"


#ifdef __cplusplus
extern "C" {
#endif

struct wmr_controller_base;

/*!
 * A connection for communicating with the controller.
 * The mechanism is implementation specific, so there are
 * two variants for either communicating directly with a
 * controller via bluetooth, and another for talking
 * to a controller through a headset tunnelled mapping.
 *
 * The controller implementation doesn't need to care how
 * the communication is implemented.
 *
 * The HMD-tunnelled version of the connection is reference
 * counted and mutex protected, as both the controller and
 * the HMD need to hold a reference to it to clean up safely.
 * For bluetooth controllers, destruction of the controller
 * xrt_device calls disconnect and destroys the connection
 * object (and bluetooth listener) immediately.
 */
struct wmr_controller_connection
{
	//! The controller this connection is talking to.
	struct wmr_controller_base *wcb;

	bool (*send_bytes)(struct wmr_controller_connection *wcc, const uint8_t *buffer, uint32_t buf_size);
	void (*receive_bytes)(struct wmr_controller_connection *wcc,
	                      uint64_t time_ns,
	                      uint8_t *buffer,
	                      uint32_t buf_size);
	int (*read_sync)(struct wmr_controller_connection *wcc, uint8_t *buffer, uint32_t buf_size, int timeout_ms);

	void (*disconnect)(struct wmr_controller_connection *wcc);
};

static inline bool
wmr_controller_connection_send_bytes(struct wmr_controller_connection *wcc, const uint8_t *buffer, uint32_t buf_size)
{
	assert(wcc->send_bytes != NULL);
	return wcc->send_bytes(wcc, buffer, buf_size);
}

static inline int
wmr_controller_connection_read_sync(struct wmr_controller_connection *wcc,
                                    uint8_t *buffer,
                                    uint32_t buf_size,
                                    int timeout_ms)
{
	return wcc->read_sync(wcc, buffer, buf_size, timeout_ms);
}

static inline void
wmr_controller_connection_disconnect(struct wmr_controller_connection *wcc)
{
	wcc->disconnect(wcc);
}

struct wmr_controller_base_imu_sample
{
	uint32_t timestamp_ticks;
	struct xrt_vec3 acc;
	struct xrt_vec3 gyro;
	int32_t temperature;
};

/*!
 * Common base for all WMR controllers.
 *
 * @ingroup drv_wmr
 * @implements xrt_device
 */
struct wmr_controller_base
{
	//! Base struct.
	struct xrt_device base;

	//! Mutex protects the controller connection
	struct os_mutex conn_lock;

	//! The connection for this controller.
	struct wmr_controller_connection *wcc;

	//! Callback from the connection when a packet has been received.
	void (*receive_bytes)(struct wmr_controller_base *wcb, uint64_t time_ns, uint8_t *buffer, uint32_t buf_size);

	enum u_logging_level log_level;

	//! Controller tracker connection that is doing 6dof tracking of this controller
	struct t_constellation_tracked_device_connection *tracking_connection;

	//! The HMD this controller is tracked relative to (its base xrt_device). Set at attach. Its live
	//! tracked pose is the reference for body-locking the controller out of camera view (so an out-of-view
	//! controller rides with the head at arm's reach) and for the room-roam-invariant arm-reach adoption
	//! gate. NULL until attached (e.g. standalone tests) — the fusion then degrades to world-frame holds.
	struct xrt_device *hmd_xdev;

	//! Mutex protects shared data used from OpenXR callbacks
	struct os_mutex data_lock;

	//! Callback to parse a controller update packet and update the input / imu info. Called with the
	//  data lock held.
	bool (*handle_input_packet)(struct wmr_controller_base *wcb,
	                            uint64_t time_ns,
	                            uint8_t *buffer,
	                            uint32_t buf_size);

	/* firmware configuration block */
	bool have_config;
	struct wmr_controller_config config;

	//! Offset for aim pose (set by subclasses. adjustable in debug)
	struct xrt_pose P_aim;

	//! Offset for grip pose relative to aim (set by subclasses. adjustable in debug)
	struct xrt_pose P_aim_grip;

	//! Thumbstick deadzone setting. Applied / used by subclasses
	float thumbstick_deadzone;

	//! Last ticks counter from input, extended to 64-bits
	uint64_t last_timestamp_ticks;

	//! Newest reported tracked pose
	struct xrt_pose pose;

	//! Time of last IMU sample, in CPU time.
	uint64_t last_imu_timestamp_ns;
	//! Last HMD-pose query used for out-of-view body-anchor fusion.
	uint64_t last_body_anchor_query_ns;
	//! Time of last IMU sample, in device time.
	uint64_t last_imu_device_timestamp_ns;
	//!< Min-Skew estimator for IMU to monotonic clock. Protected by data_lock
	struct m_clock_windowed_skew_tracker *hw2mono_clock;
	//!< Last IMU sample received
	struct wmr_controller_base_imu_sample last_imu;

	//!< Last frame timestamp in CPU mono ts notified from the tracker
	uint64_t last_frame_timestamp;
	//!< Last frame sequence number notified from the tracker
	uint64_t last_frame_sequence;

	//! Last timestamp of tracked pose from optical controller tracking
	timepoint_ns last_tracked_pose_ts;
	//! Last tracked pose from optical controller tracking
	struct xrt_pose last_tracked_pose;
	//! Last LED brightness report from optical controller tracking
	uint16_t last_brightness_report;

	//!< Command counter for timesync and keep-alives. conn_lock
	uint8_t cmd_counter;
	//!< Next (local mono) time to send keepalive. conn_lock
	uint64_t next_keepalive_timestamp_ns;
	//!< Last timesync counter. 0, 1 or 2 then loops. Starts @ 1. conn_lock
	uint8_t timesync_counter;
	//!< Variable in the timesync packet. valid values: 1..399.
	uint16_t timesync_led_intensity;
	//!< Variable in the timesync packet. valid values: 0..1023
	uint16_t timesync_val2;
	uint64_t timesync_device_slam_time_us;
	uint16_t timesync_time_offset;
	bool timesync_updated;

	//! Time of last timesync SLAM frame estimate, in CPU time.
	uint64_t last_timesync_timestamp_ns;
	//! Time of last timesync SLAM frame estimate, in device time.
	uint64_t last_timesync_device_timestamp_ns;

	struct u_var_draggable_u16 timesync_led_intensity_uvar;
	struct u_var_draggable_u16 timesync_val2_uvar;
	struct u_var_draggable_u16 timesync_time_offset_uvar;

	//! Tightly-coupled per-LED + IMU fusion: owns the reported controller pose (position + orientation).
	struct KalmanFusionInterfaceWrapper *kalman_fusion;

	//! Controller serial, kept for the per-unit IMU calibration cache (load at config-read, save at deinit).
	char imu_cal_serial[16 + 1];

	//! Constant controller-IMU vs headset-camera clock offset (ns) added to optical timestamps before
	//! fusion (default 0, env G2_CTRL_TD_NS). The OOSM handles variable processing lag; this is the
	//! residual fixed sensor-pair skew, exposed for calibration without assuming a value.
	int64_t ctrl_optical_td_ns;
};

bool
wmr_controller_base_init(struct wmr_controller_base *wcb,
                         struct wmr_controller_connection *conn,
                         enum xrt_device_type controller_type,
                         enum u_logging_level log_level,
                         u_device_destroy_function_t destroy_fn);

void
wmr_controller_base_deinit(struct wmr_controller_base *wcb);

//! Status-only packets still update the hardware clock, but only valid measurements update IMU state.
void
wmr_controller_base_imu_sample(struct wmr_controller_base *wcb,
                               struct wmr_controller_base_imu_sample *imu,
                               timepoint_ns rx_mono_ns,
                               bool imu_valid);

static inline void
wmr_controller_connection_receive_bytes(struct wmr_controller_connection *wcc,
                                        uint64_t time_ns,
                                        uint8_t *buffer,
                                        uint32_t buf_size)
{

	if (wcc->receive_bytes != NULL) {
		wcc->receive_bytes(wcc, time_ns, buffer, buf_size);
	} else {
		/* Default: deliver directly to the controller instance */
		struct wmr_controller_base *wcb = wcc->wcb;
		assert(wcb->receive_bytes != NULL);
		wcb->receive_bytes(wcb, time_ns, buffer, buf_size);
	}
}

static inline struct xrt_device *
wmr_controller_base_to_xrt_device(struct wmr_controller_base *wcb)
{
	if (wcb != NULL) {
		return &wcb->base;
	}
	return NULL;
}

/* Tell the controller which HMD to register with for tracking */
void
wmr_controller_attach_to_hmd(struct wmr_controller_base *wcb, struct wmr_hmd *hmd);

#ifdef __cplusplus
}
#endif
