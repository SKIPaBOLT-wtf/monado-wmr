// Copyright 2018, Philipp Zabel.
// Copyright 2020-2021, N Madsen.
// Copyright 2020-2023, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Interface to the WMR HMD driver code.
 * @author Philipp Zabel <philipp.zabel@gmail.com>
 * @author nima01 <nima_zero_one@protonmail.com>
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @author Nova King <technobaboo@proton.me>
 * @ingroup drv_wmr
 */

#pragma once

#include "os/os_threading.h"
#include "math/m_imu_3dof.h"
#include "tracking/t_tracking.h"
#include "tracking/t_constellation_tracking.h"
#include "util/u_logging.h"
#include "util/u_distortion_mesh.h"
#include "util/u_var.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_frame.h"
#include "xrt/xrt_prober.h"

#include "wmr_protocol.h"
#include "wmr_config.h"
#include "wmr_camera.h"
#include "wmr_common.h"
#include "wmr_hmd_controller.h"


#ifdef __cplusplus
extern "C" {
#endif

/* Support 2 controllers on HP Reverb G2 */
#define WMR_MAX_CONTROLLERS 2

/* Only the HMD reader owns this queue. Firmware waits must not recursively create controllers.
 * If it fills, the current initialization fails explicitly before reading another HID report. */
#define WMR_PENDING_REPORT_CAPACITY 128
struct wmr_pending_report
{
	uint64_t received_ns;
	uint32_t size;
	uint8_t data[WMR_FEATURE_BUFFER_SIZE];
};

struct wmr_hmd;

//! Optional diagnostics, owned exclusively by the sensor reading thread.
struct wmr_imu_diagnostics
{
	uint64_t window_start_ns;
	uint64_t next_window_ns;
	uint64_t sample_count;
	double accel_mean;
	double accel_m2;
	double accel_min;
	double accel_max;
	double gyro_mean;
	double gyro_max;
};

struct wmr_headset_descriptor
{
	enum wmr_headset_type hmd_type;

	//! String by which we recognise the device
	const char *dev_id_str;
	//! Friendly ID string for debug
	const char *debug_name;

	int (*init_func)(struct wmr_hmd *wh);
	void (*deinit_func)(struct wmr_hmd *wh);
	void (*screen_enable_func)(struct wmr_hmd *wh, bool enable);
};

/*!
 * @implements xrt_device
 */
struct wmr_hmd
{
	struct xrt_device base;

	const struct wmr_headset_descriptor *hmd_desc;

	//! firmware configuration block, with device names etc
	struct wmr_config_header config_hdr;

	//! Config data parsed from the firmware JSON
	struct wmr_hmd_config config;

	//! Absolute accel scale (LSB -> m/s^2): per-device calibration. Per-headset-type
	//! default, refined by wmr/accel-scale.json in the config dir.
	float accel_scale;

	struct wmr_imu_diagnostics imu_diagnostics;

	//! Packet reading thread.
	struct os_thread_helper oth;

	enum u_logging_level log_level;

	/*!
	 * This is the Hololens Sensors device, this is where we get all of the
	 * IMU data and read the config from.
	 *
	 * During start it is owned by the thread creating the device, after
	 * init it is owned by the reading thread. Read/write access is
	 * protected by the hid_lock
	 */

	struct os_hid_device *hid_hololens_sensors_dev;
	struct os_mutex hid_lock;

	/*!
	 * This is the vendor specific companion device of the Hololens Sensors.
	 * When activated, it will report the physical IPD adjustment and proximity
	 * sensor status of the headset. It also allows enabling/disabling the HMD
	 * screen on Reverb G1/G2.
	 */
	struct os_hid_device *hid_control_dev;

	//! Current desired HMD screen state.
	bool hmd_screen_enable;
	//! Latest raw IPD value read from the device.
	uint16_t raw_ipd;
	//! Latest proximity sensor value read from the device.
	uint8_t proximity_sensor;

	struct hololens_sensors_packet packet;

	struct
	{
		//! Protects all members of the `fusion` substruct.
		struct os_mutex mutex;

		//! Main fusion calculator.
		struct m_imu_3dof i3dof;

		//! The last angular velocity from the IMU, for prediction.
		struct xrt_vec3 last_angular_velocity;

		//! When did we get the last IMU sample, in CPU time.
		uint64_t last_imu_timestamp_ns;
	} fusion;

	//! Fields related to camera-based tracking (SLAM and hand tracking)
	struct
	{
		//! Source of video/IMU data for tracking
		struct xrt_fs *source;

		//! Context for @ref source
		struct xrt_frame_context xfctx;

		//! SLAM tracker.
		//! @todo Right now, we are not consistent in how we interface with
		//! trackers. In particular, we have a @ref xrt_tracked_slam field but not
		//! an equivalent for hand tracking.
		struct xrt_tracked_slam *slam;

		struct xrt_slam_sinks *slam_sinks;

		//! Calibration data for SLAM
		struct t_slam_calibration slam_calib;

		//! Calibration data for constellation tracking
		struct t_constellation_camera_group constellation_calib;

		//! Set at start. Whether the SLAM tracker was initialized.
		bool slam_enabled;

		//! Set at start. Whether the hand tracker was initialized.
		bool hand_enabled;

		//! SLAM systems track the IMU pose, enabling this corrects it to middle of the eyes
		bool imu2me;
	} tracking;

	//! Whether to track the HMD with 6dof SLAM or fallback to the `fusion` 3dof tracker
	bool slam_over_3dof;

	//! Last tracked pose
	struct xrt_pose pose;

	//! Additional offset to apply to `pose`
	struct xrt_pose offset;

	//! Average 4 IMU samples before sending them to the trackers
	bool average_imus;

	/*!
	 * Offset for tracked pose offsets (applies to both fusion and SLAM).
	 * Applied when getting the tracked poses, so is effectively a offset
	 * to increase or decrease prediction.
	 */
	struct u_var_draggable_f32 tracked_offset_ms;

	struct
	{
		struct u_var_button hmd_screen_enable_btn;
		struct u_var_button switch_tracker_btn;
		char hand_status[128];
		char slam_status[128];
	} gui;

	/* Tunnelled controller devices (Reverb G2, Odyssey+) handling */
	struct os_mutex controller_status_lock;
	struct os_cond controller_status_cond;
	bool have_left_controller_status;
	bool have_right_controller_status;

	struct wmr_hmd_controller_connection *controller[WMR_MAX_CONTROLLERS];
	struct wmr_pending_report pending_reports[WMR_PENDING_REPORT_CAPACITY];
	uint32_t pending_report_head;
	uint32_t pending_report_count;

	struct t_constellation_tracker *controller_tracker;
};

static inline struct wmr_hmd *
wmr_hmd(struct xrt_device *p)
{
	return (struct wmr_hmd *)p;
}

static inline struct xrt_device *
wmr_hmd_to_xrt_device(struct wmr_hmd *hmd)
{
	if (hmd) {
		return &hmd->base;
	}
	return NULL;
}

void
wmr_hmd_create(enum wmr_headset_type hmd_type,
               struct os_hid_device *hid_holo,
               struct os_hid_device *hid_ctrl,
               struct xrt_prober_device *dev_holo,
               enum u_logging_level log_level,
               struct wmr_hmd **out_hmd,
               struct xrt_device **out_handtracker,
               struct wmr_controller_base **out_left_controller,
               struct wmr_controller_base **out_right_controller);

bool
wmr_hmd_send_controller_packet(struct wmr_hmd *hmd, const uint8_t *buffer, uint32_t buf_size);
int
wmr_hmd_read_sync_from_controller(struct wmr_hmd *hmd,
                                  uint8_t hmd_cmd_base,
                                  uint8_t *buffer,
                                  uint32_t buf_size,
                                  int timeout_ms);

struct t_constellation_tracked_device_connection *
wmr_hmd_add_tracked_controller(struct wmr_hmd *hmd,
                               struct xrt_device *xdev,
                               struct t_constellation_tracked_device_callbacks *cb);

#ifdef __cplusplus
}
#endif
