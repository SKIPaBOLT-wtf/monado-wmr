// Copyright 2020-2021, N Madsen.
// Copyright 2020-2023, Collabora, Ltd.
// Copyright 2020-2023, Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Driver for WMR Controller.
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */

#include "os/os_time.h"
#include "os/os_hid.h"

#include "math/m_mathinclude.h"
#include "math/m_api.h"
#include "math/m_vec2.h"
#include "math/m_vec3.h"
#include "math/m_predict.h"
#include "math/m_space.h"

#include "tracking/t_constellation_tracking.h"

#include "util/u_file.h"
#include "util/u_var.h"
#include "util/u_misc.h"
#include "util/u_time.h"
#include "util/u_debug.h"
#include "util/u_device.h"
#include "util/u_g2_telemetry.h"
#include "util/u_trace_marker.h"

#include "wmr_common.h"
#include "wmr_controller_base.h"
#include "wmr_g2_timing.h"
#include "wmr_config_key.h"
#include "wmr_hmd.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <inttypes.h>

#define WMR_TRACE(wcb, ...) U_LOG_XDEV_IFL_T(&wcb->base, wcb->log_level, __VA_ARGS__)
#define WMR_TRACE_HEX(wcb, ...) U_LOG_XDEV_IFL_T_HEX(&wcb->base, wcb->log_level, __VA_ARGS__)
#define WMR_DEBUG(wcb, ...) U_LOG_XDEV_IFL_D(&wcb->base, wcb->log_level, __VA_ARGS__)
#define WMR_DEBUG_HEX(wcb, ...) U_LOG_XDEV_IFL_D_HEX(&wcb->base, wcb->log_level, __VA_ARGS__)
#define WMR_INFO(wcb, ...) U_LOG_XDEV_IFL_I(&wcb->base, wcb->log_level, __VA_ARGS__)
#define WMR_WARN(wcb, ...) U_LOG_XDEV_IFL_W(&wcb->base, wcb->log_level, __VA_ARGS__)
#define WMR_ERROR(wcb, ...) U_LOG_XDEV_IFL_E(&wcb->base, wcb->log_level, __VA_ARGS__)
#define WMR_BODY_ANCHOR_QUERY_PERIOD_NS (33ULL * 1000ULL * 1000ULL)

#define wmr_controller_hexdump_buffer(wcb, label, buf, length)                                                         \
	do {                                                                                                           \
		WMR_DEBUG(wcb, "%s", label);                                                                           \
		WMR_DEBUG_HEX(wcb, buf, length);                                                                       \
	} while (0);


static inline struct wmr_controller_base *
wmr_controller_base(struct xrt_device *p)
{
	return (struct wmr_controller_base *)p;
}

/* Telemetry device_id for a controller: 1=left, 2=right (0 reserved for HMD). */
static inline uint8_t
wmr_controller_telem_id(struct wmr_controller_base *wcb)
{
	return (wcb->base.device_type == XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER) ? 2 : 1;
}

/* Called from subclasses' handle_input_packet method, with data_lock */
void
wmr_controller_base_imu_sample(struct wmr_controller_base *wcb,
                               struct wmr_controller_base_imu_sample *imu_sample,
                               timepoint_ns rx_mono_ns,
                               bool imu_valid)
{
	/* Extend 32-bit tick count to 64-bit and convert to ns */
	uint32_t tick_delta = imu_sample->timestamp_ticks - (uint32_t)wcb->last_timestamp_ticks;
	wcb->last_timestamp_ticks += tick_delta;

	timepoint_ns now_hw_ns = wcb->last_timestamp_ticks * WMR_MOTION_CONTROLLER_NS_PER_TICK;

	/* Update windowed min-skew estimator and convert hardware timestamp into monotonic clock */
	m_clock_windowed_skew_tracker_push(wcb->hw2mono_clock, rx_mono_ns, now_hw_ns);

	timepoint_ns mono_time_ns;
	if (!m_clock_windowed_skew_tracker_to_local(wcb->hw2mono_clock, now_hw_ns, &mono_time_ns)) {
		WMR_DEBUG(wcb,
		          "Dropping IMU sample until clock estimator synchronises. Rcv ts %" PRIu64 " hw ts %" PRIu64,
		          rx_mono_ns, now_hw_ns);
		return;
	}

	// Preserve the distinction between a hardware sleep status and a real sensor sample.
	// The fusion can hold a recently confirmed rest state without inventing an IMU measurement.
	if (!imu_valid) {
		kalman_fusion_process_idle_status(wcb->kalman_fusion, mono_time_ns);
		return;
	}

	/*
	 * Check if the timepoint does time travel, we get one or two
	 * old samples when the device has not been cleanly shut down,
	 * and if the controller is left idle and goes into low power
	 * mode it can come back with a different clock epoch
	 */
	if (wcb->last_imu_timestamp_ns > (uint64_t)mono_time_ns) {
		/* Backwards-in-time delta, in the same monotonic domain on both sides. */
		int64_t back_ns = (int64_t)wcb->last_imu_timestamp_ns - (int64_t)mono_time_ns;
		WMR_WARN(wcb,
		         "Received sample from the past, new: %" PRIu64 ", last: %" PRIu64 ", diff: %" PRId64
		         " ns. resetting clock tracking",
		         mono_time_ns, wcb->last_imu_timestamp_ns, back_ns);
		/* Telemetry: IMU went backwards in time -> anomaly (fusion is being reset).
		 * value is the backwards delta in MILLISECONDS: an ns delta does not fit
		 * f32's ~24-bit mantissa for large clock jumps; ms keeps it meaningful.
		 * The schema documents value's unit for this event type. */
		g2_telem_event(wmr_controller_telem_id(wcb), (uint64_t)mono_time_ns, G2_TELEM_EV_IMU_ANOMALY,
		               (float)((double)back_ns / 1.0e6));
		// Drop this sample and reset clock tracking so the next sample re-establishes the epoch.
		// The filter keeps its state (it tolerates the gap; a non-monotonic sample is never fed to it).
		wcb->last_imu_timestamp_ns = 0;
		wcb->last_imu_device_timestamp_ns = 0;
		m_clock_windowed_skew_tracker_reset(wcb->hw2mono_clock);
		m_clock_windowed_skew_tracker_push(wcb->hw2mono_clock, rx_mono_ns, now_hw_ns);
		return;
	}

	float accel_m_p_s_2 = m_vec3_len(imu_sample->acc);
	WMR_TRACE(wcb, "Accel [m/s^2] : %f", accel_m_p_s_2);

	// if it accelerates quite quickly, then we up the brightness to make it easier to find constellation poses
	if (accel_m_p_s_2 > 20) {
		wcb->timesync_led_intensity = MIN(wcb->timesync_led_intensity + 10, 399);
	}

	wcb->last_imu_timestamp_ns = mono_time_ns;
	wcb->last_imu_device_timestamp_ns = now_hw_ns;
	wcb->last_imu = *imu_sample;

	/* Telemetry: controller IMU sample. hw_ts_ns uses mono_time_ns, the monotonic-
	 * converted sample time, so it shares the common clock with the HMD IMU and every
	 * other stream (no raw device ticks in hw_ts_ns). */
	g2_telem_imu(wmr_controller_telem_id(wcb), (uint64_t)mono_time_ns, imu_sample->acc.x, imu_sample->acc.y,
	             imu_sample->acc.z, imu_sample->gyro.x, imu_sample->gyro.y, imu_sample->gyro.z);

	struct xrt_imu_sample k_imu_sample = {
	    .timestamp_ns = mono_time_ns,
	    .gyro_rad_secs = {imu_sample->gyro.x, imu_sample->gyro.y, imu_sample->gyro.z},
	    .accel_m_s2 = {imu_sample->acc.x, imu_sample->acc.y, imu_sample->acc.z}};
	// Pass NULL variances so the fusion uses its tuned, test-validated
	// internal defaults — keeping the real driver's behaviour identical to
	// what the t_tracker_kalman_fusion test suite verifies.
	kalman_fusion_process_imu_data(wcb->kalman_fusion, &k_imu_sample, NULL, NULL);

}

static void
wmr_controller_base_send_timesync(struct wmr_controller_base *wcb);
static void
wmr_controller_base_send_keepalive(struct wmr_controller_base *wcb, uint64_t time_ns);

static void
receive_bytes(struct wmr_controller_base *wcb, uint64_t time_ns, uint8_t *buffer, uint32_t buf_size)
{
	if (buf_size < 1) {
		WMR_ERROR(wcb, "WMR Controller: Error receiving short packet");
		return;
	}

	switch (buffer[0]) {
	case WMR_MOTION_CONTROLLER_STATUS_MSG: {
		uint64_t body_anchor_query_ns = 0;
		os_mutex_lock(&wcb->data_lock);
		// Send a timesync packet if needed
		if (wcb->timesync_updated) {
			wmr_controller_base_send_timesync(wcb);
			wcb->timesync_updated = false;
		}

		wmr_controller_base_send_keepalive(wcb, time_ns);

		// Note: skipping msg type byte
		bool b = wcb->handle_input_packet(wcb, time_ns, &buffer[1], (size_t)buf_size - 1);
		body_anchor_query_ns = wcb->last_imu_timestamp_ns;
		os_mutex_unlock(&wcb->data_lock);

		if (!b) {
			WMR_ERROR(wcb, "WMR Controller: Failed handling message type: %02x, size: %i", buffer[0],
			          buf_size);
			wmr_controller_hexdump_buffer(wcb, "Controller Message", buffer, buf_size);
			return;
		}

		// Out-of-view body anchor: query the live HMD pose outside data_lock. The IMU packet parser holds
		// data_lock at controller packet rate, while xrt_device_get_tracked_pose may recurse into the HMD
		// tracker; doing that under the controller lock can create avoidable lock contention and jitter.
		if (wcb->hmd_xdev != NULL && body_anchor_query_ns != 0 &&
		    (wcb->last_body_anchor_query_ns == 0 ||
		     body_anchor_query_ns < wcb->last_body_anchor_query_ns ||
		     body_anchor_query_ns - wcb->last_body_anchor_query_ns >= WMR_BODY_ANCHOR_QUERY_PERIOD_NS)) {
			wcb->last_body_anchor_query_ns = body_anchor_query_ns;
			struct xrt_space_relation hmd_rel;
			if (xrt_device_get_tracked_pose(wcb->hmd_xdev, XRT_INPUT_GENERIC_TRACKER_POSE,
			                                body_anchor_query_ns, &hmd_rel) == XRT_SUCCESS &&
			    (hmd_rel.relation_flags & XRT_SPACE_RELATION_POSITION_VALID_BIT) != 0) {
				kalman_fusion_update_body_anchor(wcb->kalman_fusion, &hmd_rel.pose);
			}
		}

		break;
	}
	default: WMR_DEBUG(wcb, "WMR Controller: Unknown message type: %02x, size: %i", buffer[0], buf_size); break;
	}

	return;
}

static bool
wmr_controller_send_bytes(struct wmr_controller_base *wcb, const uint8_t *buffer, uint32_t buf_size)
{
	bool res = false;

	os_mutex_lock(&wcb->conn_lock);
	struct wmr_controller_connection *conn = wcb->wcc;
	if (conn != NULL) {
		res = wmr_controller_connection_send_bytes(conn, buffer, buf_size);
	}
	os_mutex_unlock(&wcb->conn_lock);

	return res;
}

static int
wmr_controller_read_sync(struct wmr_controller_base *wcb, uint8_t *buffer, uint32_t buf_size, int timeout_ms)
{
	int res = -1;
	os_mutex_lock(&wcb->conn_lock);
	struct wmr_controller_connection *conn = wcb->wcc;
	if (conn != NULL) {
		res = wmr_controller_connection_read_sync(conn, buffer, buf_size, timeout_ms);
	}
	os_mutex_unlock(&wcb->conn_lock);

	return res;
}

static int
wmr_controller_send_fw_cmd(struct wmr_controller_base *wcb,
                           const struct wmr_controller_fw_cmd *fw_cmd,
                           unsigned char response_code,
                           struct wmr_controller_fw_cmd_response *response)
{
	// comms timeout. Replies are usually in 10ms or so but the first can take longer
	const int timeout_ms = 250;
	const int timeout_ns = timeout_ms * U_TIME_1MS_IN_NS;
	int64_t timeout_start = os_monotonic_get_ns();
	int64_t timeout_end_ns = timeout_start + timeout_ns;

	if (!wmr_controller_send_bytes(wcb, fw_cmd->buf, sizeof(fw_cmd->buf))) {
		return -1;
	}

	do {
		int64_t remaining_ns = timeout_end_ns - os_monotonic_get_ns();
		if (remaining_ns <= 0) {
			break;
		}
		int remaining_ms = (int)((remaining_ns + U_TIME_1MS_IN_NS - 1) / U_TIME_1MS_IN_NS);
		int size = wmr_controller_read_sync(wcb, response->buf, sizeof(response->buf), remaining_ms);
		if (size == -1) {
			return -1;
		}

		if (size < 1) {
			// Ignore 0-byte reads (timeout) and try again
			continue;
		}

		if (response->buf[0] == response_code) {
			WMR_TRACE(wcb, "Controller fw read returned %d bytes", size);
			if (size != sizeof(response->buf) || (response->response.cmd_id_echo != fw_cmd->cmd.cmd_id)) {
				WMR_DEBUG(
				    wcb, "Unexpected fw response - size %d (expected %zu), cmd_id_echo %u != cmd_id %u",
				    size, sizeof(response->buf), response->response.cmd_id_echo, fw_cmd->cmd.cmd_id);
				return -1;
			}

			response->response.blk_remain = __le32_to_cpu(response->response.blk_remain);
			return size;
		}
	} while (os_monotonic_get_ns() < timeout_end_ns);

	WMR_WARN(wcb, "Controller fw read timed out after %u ms",
	         (unsigned int)((os_monotonic_get_ns() - timeout_start) / U_TIME_1MS_IN_NS));
	return -ETIMEDOUT;
}

XRT_MAYBE_UNUSED static int
wmr_read_fw_block(struct wmr_controller_base *d, uint8_t blk_id, uint8_t **out_data, size_t *out_size)
{
	struct wmr_controller_fw_cmd_response fw_cmd_response;

	uint8_t *data;
	uint8_t *data_pos;
	uint8_t *data_end;
	uint32_t data_size;
	uint32_t remain;

	struct wmr_controller_fw_cmd fw_cmd;
	memset(&fw_cmd, 0, sizeof(fw_cmd));

	fw_cmd = WMR_CONTROLLER_FW_CMD_INIT(0x06, 0x02, blk_id, 0xffffffff);
	if (wmr_controller_send_fw_cmd(d, &fw_cmd, 0x02, &fw_cmd_response) < 0) {
		WMR_WARN(d, "Failed to read fw - cmd 0x02 failed to read header for block %d", blk_id);
		return -1;
	}

	data_size = fw_cmd_response.response.blk_remain + fw_cmd_response.response.len;
	WMR_DEBUG(d, "FW header %d bytes, %u bytes in block", fw_cmd_response.response.len, data_size);
	if (data_size == 0)
		return -1;

	data = calloc(1, data_size + 1);
	if (!data) {
		return -1;
	}
	data[data_size] = '\0';

	remain = data_size;
	data_pos = data;
	data_end = data + data_size;

	uint8_t to_copy = fw_cmd_response.response.len;

	memcpy(data_pos, fw_cmd_response.response.data, to_copy);
	data_pos += to_copy;
	remain -= to_copy;

	while (remain > 0) {
		fw_cmd = WMR_CONTROLLER_FW_CMD_INIT(0x06, 0x02, blk_id, remain);

		os_nanosleep(U_TIME_1MS_IN_NS * 10); // Sleep 10ms
		if (wmr_controller_send_fw_cmd(d, &fw_cmd, 0x02, &fw_cmd_response) < 0) {
			WMR_WARN(d, "Failed to read fw - cmd 0x02 failed @ offset %zu", data_pos - data);
			return -1;
		}

		uint8_t to_copy = fw_cmd_response.response.len;
		if (data_pos + to_copy > data_end)
			to_copy = data_end - data_pos;

		WMR_DEBUG(d, "Read %d bytes @ offset %zu / %d", to_copy, data_pos - data, data_size);
		memcpy(data_pos, fw_cmd_response.response.data, to_copy);
		data_pos += to_copy;
		remain -= to_copy;
	}

	WMR_DEBUG(d, "Read %d-byte FW data block %d", data_size, blk_id);
	wmr_controller_hexdump_buffer(d, "Data block", data, data_size);

	*out_data = data;
	*out_size = data_size;

	return 0;
}

/*
 *
 * Config functions.
 *
 */
static bool
read_controller_fw_info(struct wmr_controller_base *wcb,
                        uint32_t *fw_revision,
                        uint16_t *calibration_size,
                        char serial_no[16])
{
	uint8_t *data = NULL;
	size_t data_size;
	int ret;

	/* FW block 0 contains the FW revision (offset 0x14, size 4) and
	 * calibration block size (offset 0x34 size 2) */
	ret = wmr_read_fw_block(wcb, 0x0, &data, &data_size);
	if (ret < 0 || data == NULL) {
		WMR_ERROR(wcb, "Failed to read FW info block 0");
		return false;
	}
	if (data_size < 0x36) {
		WMR_ERROR(wcb, "Failed to read FW info block 0 - too short");
		free(data);
		return false;
	}


	const unsigned char *tmp = data + 0x14;
	*fw_revision = read32(&tmp);
	tmp = data + 0x34;
	*calibration_size = read16(&tmp);

	free(data);

	/* FW block 3 contains the controller serial number at offset
	 * 0x84, size 16 bytes */
	ret = wmr_read_fw_block(wcb, 0x3, &data, &data_size);
	if (ret < 0 || data == NULL) {
		WMR_ERROR(wcb, "Failed to read FW info block 3");
		return false;
	}
	if (data_size < 0x94) {
		WMR_ERROR(wcb, "Failed to read FW info block 3 - too short");
		free(data);
		return false;
	}

	memcpy(serial_no, data + 0x84, 0x10);
	serial_no[16] = '\0';

	free(data);
	return true;
}

char *
build_cache_filename(char *serial_no)
{
	int outlen = strlen("controller-") + strlen(serial_no) + strlen(".json") + 1;
	char *out = malloc(outlen);
	int ret = snprintf(out, outlen, "controller-%s.json", serial_no);

	assert(ret <= outlen);
	(void)ret;

	// Make sure the filename is valid
	for (char *cur = out; *cur != '\0'; cur++) {
		if (!isalnum(*cur) && *cur != '.') {
			*cur = '_';
		}
	}

	return out;
}

static bool
read_calibration_cache(struct wmr_controller_base *wcb, char *cache_filename)
{
	FILE *f = u_file_open_file_in_config_dir_subpath("wmr", cache_filename, "r");
	uint8_t *buffer = NULL;

	if (f == NULL) {
		WMR_DEBUG(wcb, "Failed to open wmr/%s cache file or it doesn't exist.", cache_filename);
		return false;
	}

	// Read the file size to allocate a read buffer
	fseek(f, 0L, SEEK_END);
	size_t file_size = ftell(f);

	// Reset and read the data
	fseek(f, 0L, SEEK_SET);

	buffer = calloc(file_size + 1, sizeof(uint8_t));
	if (buffer == NULL) {
		goto fail;
	}
	buffer[file_size] = '\0';

	size_t ret = fread(buffer, sizeof(char), file_size, f);
	if (ret != file_size) {
		WMR_WARN(wcb, "Cache file wmr/%s failed to read %u bytes (got %u)", cache_filename, (int)file_size,
		         (int)ret);
		goto fail;
	}

	if (!wmr_controller_config_parse(&wcb->config, (char *)buffer, wcb->log_level)) {
		WMR_WARN(wcb, "Cache file wmr/%s contains invalid JSON. Ignoring", cache_filename);
		goto fail;
	}

	fclose(f);
	free(buffer);

	return true;

fail:
	if (buffer) {
		free(buffer);
	}
	fclose(f);
	return false;
}

static void
write_calibration_cache(struct wmr_controller_base *wcb, char *cache_filename, uint8_t *data, size_t data_size)
{
	FILE *f = u_file_open_file_in_config_dir_subpath("wmr", cache_filename, "w");
	if (f == NULL) {
		return;
	}

	size_t ret = fwrite(data, sizeof(char), data_size, f);
	if (ret != data_size) {
		fclose(f);
		return;
	}

	fclose(f);
}

/* Cross-session IMU calibration cache (per controller serial). The converged gyro/accel bias + accel
 * scale are quasi-constant per unit, so persisting them and seeding the next session makes the first
 * second accurate (before any stance) and survives a large physical bias. The optically-derived
 * INTRINSICS (gyro scale + gyro->device misalignment as one 3x3 M_g, and the accel ellipsoid T_a, both
 * computed offline by imu_calib_from_optical.py) live here too — quasi-constant per unit, applied from
 * the first sample. Plain text, one line; v2 appends the two 3x3 matrices to the v1 layout (a v1 file
 * still loads — the matrices default to identity = uncorrected):
 *   version  bg_x bg_y bg_z  ba_x ba_y ba_z  scale  count  M_g[9 row-major]  T_a[9 row-major]
 */
#define IMU_CAL_VERSION 2

/* Identity 3x3, row-major: the "no correction" default for the intrinsics matrices. */
static const double IMU_CAL_IDENTITY3[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

static void
imu_cal_filename(const char *serial, char *out, size_t out_size)
{
	snprintf(out, out_size, "imu-cal-%s.txt", serial);
	for (char *c = out; *c != '\0'; c++) {
		if (!isalnum(*c) && *c != '.' && *c != '-' && *c != '_') {
			*c = '_';
		}
	}
}

/* MEMS-plausible intrinsics: ICM-20602 sensitivity tolerance is ~±3% with <2% cross-axis, so beyond
 * 8% diagonal / 0.06 off-diagonal is a degenerate fit, not hardware (a non-gravity-spanning rest set
 * once persisted a 17.6% accel x-scale that measurably corrupted live tracking). */
static bool
imu_cal_intrinsics_plausible(const double m[9])
{
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 3; c++) {
			const double v = m[r * 3 + c];
			if (!isfinite(v) || (r == c ? fabs(v - 1.0) > 0.08 : fabs(v) > 0.06)) {
				return false;
			}
		}
	}
	return true;
}

/* Parse a cached record (v1 or v2). Returns true on a well-formed, PLAUSIBLE bias/scale; fills @p mg /
 * @p ta with the v2 intrinsics, or identity for a v1 record (no intrinsics) or implausible matrices
 * (@p intrinsics_rejected set). Bounds mirror the filter's plausibility caps: real MEMS gyro bias
 * << 0.1 rad/s, accel bias << 0.5 m/s^2, scale within ~10%. */
static bool
imu_cal_parse(FILE *f,
              double bg[3],
              double ba[3],
              double *scale,
              int *count,
              double mg[9],
              double ta[9],
              bool *intrinsics_rejected)
{
	int ver = 0;
	memcpy(mg, IMU_CAL_IDENTITY3, sizeof(IMU_CAL_IDENTITY3));
	memcpy(ta, IMU_CAL_IDENTITY3, sizeof(IMU_CAL_IDENTITY3));
	int n = fscanf(f, "%d %lf %lf %lf %lf %lf %lf %lf %d", &ver, &bg[0], &bg[1], &bg[2], &ba[0], &ba[1],
	               &ba[2], scale, count);
	/* isfinite: fscanf %lf parses "nan", and NaN sails through every magnitude comparison below
	 * (all compare false). A NaN scale is the worst case: reset_filter never touches m_accel_scale,
	 * so the filter NaNs and resets on every sample, session after session, until the cache file is
	 * hand-deleted. Reject the record instead (the caller WARNs and recalibrates). */
	if (n != 9 || (ver != 1 && ver != 2) || !isfinite(*scale) || *scale <= 0.9 || *scale >= 1.1) {
		return false;
	}
	for (int i = 0; i < 3; i++) {
		if (!isfinite(bg[i]) || !isfinite(ba[i]) || fabs(bg[i]) > 0.1 || fabs(ba[i]) > 0.5) {
			return false;
		}
	}
	if (ver == 2) {
		for (int i = 0; i < 9; i++) {
			if (fscanf(f, "%lf", &mg[i]) != 1) {
				return false;
			}
		}
		for (int i = 0; i < 9; i++) {
			if (fscanf(f, "%lf", &ta[i]) != 1) {
				return false;
			}
		}
		if (!imu_cal_intrinsics_plausible(mg) || !imu_cal_intrinsics_plausible(ta)) {
			memcpy(mg, IMU_CAL_IDENTITY3, sizeof(IMU_CAL_IDENTITY3));
			memcpy(ta, IMU_CAL_IDENTITY3, sizeof(IMU_CAL_IDENTITY3));
			if (intrinsics_rejected != NULL) {
				*intrinsics_rejected = true;
			}
		}
	}
	return true;
}

static void
imu_cal_load(struct wmr_controller_base *wcb, const char *serial)
{
	if (wcb->kalman_fusion == NULL || serial[0] == '\0') {
		return;
	}
	char fn[64];
	imu_cal_filename(serial, fn, sizeof(fn));
	FILE *f = u_file_open_file_in_config_dir_subpath("wmr", fn, "r");
	if (f == NULL) {
		return;
	}
	int count = 0;
	double bg[3], ba[3], scale = 1.0, mg[9], ta[9];
	bool intrinsics_rejected = false;
	bool plausible = imu_cal_parse(f, bg, ba, &scale, &count, mg, ta, &intrinsics_rejected);
	fclose(f);
	if (plausible) {
		kalman_fusion_set_imu_calibration(wcb->kalman_fusion, bg, ba, scale);
		kalman_fusion_set_imu_intrinsics(wcb->kalman_fusion, mg, ta); // identity for a v1 cache (no-op)
		WMR_INFO(wcb, "Loaded IMU calibration prior (serial %s, accel scale %.4f, %d sessions)", serial,
		         scale, count);
		if (intrinsics_rejected) {
			WMR_WARN(wcb, "Rejected implausible cached IMU intrinsics (serial %s) - using identity",
			         serial);
		}
	} else {
		WMR_WARN(wcb, "Ignoring implausible IMU calibration cache (serial %s) - will recalibrate", serial);
	}
}

static void
imu_cal_save(struct wmr_controller_base *wcb, const char *serial)
{
	if (wcb->kalman_fusion == NULL || serial[0] == '\0') {
		return;
	}
	double bg[3], ba[3], scale = 1.0;
	if (!kalman_fusion_get_imu_calibration(wcb->kalman_fusion, bg, ba, &scale)) {
		return; // no calibrated stance this session -> estimate not trustworthy, keep the old cache
	}
	char fn[64];
	imu_cal_filename(serial, fn, sizeof(fn));

	/* Read the prior cache once: its bias/scale to EMA-blend, and its intrinsics as the carry-forward
	 * default (so a session that did not re-fit them keeps the offline-computed matrices intact). */
	double obg[3] = {0, 0, 0}, oba[3] = {0, 0, 0}, oscale = 1.0, mg[9], ta[9];
	int ocount = 0;
	FILE *rf = u_file_open_file_in_config_dir_subpath("wmr", fn, "r");
	bool had_cache = false;
	if (rf != NULL) {
		had_cache = imu_cal_parse(rf, obg, oba, &oscale, &ocount, mg, ta, NULL);
		fclose(rf);
	}
	if (!had_cache) {
		ocount = 0;
		memcpy(mg, IMU_CAL_IDENTITY3, sizeof(IMU_CAL_IDENTITY3));
		memcpy(ta, IMU_CAL_IDENTITY3, sizeof(IMU_CAL_IDENTITY3));
	}

	/* EMA-blend the session BIAS/scale estimate into the existing cache (robust to one bad session;
	 * tracks slow hardware drift over the device's life). First time, take the session value outright. */
	const double a = (ocount > 0) ? 0.25 : 1.0;
	for (int i = 0; i < 3; i++) {
		bg[i] = (1.0 - a) * obg[i] + a * bg[i];
		ba[i] = (1.0 - a) * oba[i] + a * ba[i];
	}
	scale = (1.0 - a) * oscale + a * scale;

	/* INTRINSICS: prefer the live filter's currently-applied matrices (the offline-seeded ones, or a
	 * fresh online accel-ellipsoid fit) over the cached carry-forward. Not EMA-blended — they are a
	 * geometric correction, not a slowly-drifting bias; get_* returns false (matrices unchanged) when
	 * the filter holds no real correction, so the cached values carry forward untouched. */
	double cur_mg[9], cur_ta[9];
	if (kalman_fusion_get_imu_intrinsics(wcb->kalman_fusion, cur_mg, cur_ta) &&
	    imu_cal_intrinsics_plausible(cur_mg) && imu_cal_intrinsics_plausible(cur_ta)) {
		memcpy(mg, cur_mg, sizeof(mg));
		memcpy(ta, cur_ta, sizeof(ta));
	}

	FILE *wf = u_file_open_file_in_config_dir_subpath("wmr", fn, "w");
	if (wf != NULL) {
		fprintf(wf, "%d %.9g %.9g %.9g %.9g %.9g %.9g %.9g %d", IMU_CAL_VERSION, bg[0], bg[1], bg[2], ba[0],
		        ba[1], ba[2], scale, ocount + 1);
		for (int i = 0; i < 9; i++) {
			fprintf(wf, " %.9g", mg[i]);
		}
		for (int i = 0; i < 9; i++) {
			fprintf(wf, " %.9g", ta[i]);
		}
		fprintf(wf, "\n");
		fclose(wf);
		WMR_INFO(wcb, "Saved IMU calibration (serial %s, accel scale %.4f)", serial, scale);
	}
}

static bool
read_controller_config(struct wmr_controller_base *wcb)
{
	unsigned char *config_json_block;
	int ret;
	uint32_t fw_revision;
	uint16_t calibration_size;
	char serial_no[16 + 1];

	if (!read_controller_fw_info(wcb, &fw_revision, &calibration_size, serial_no)) {
		return false;
	}

	WMR_INFO(wcb, "Reading configuration for controller serial %s. FW revision %x", serial_no, fw_revision);

#if 0
  /* WMR also reads block 0x14, which seems to have some FW revision info,
   * but we don't use it */
	// Read block 0x14
	ret = wmr_read_fw_block(wcb, 0x14, &data, &data_size);
	if (ret < 0 || data == NULL)
		return false;
	free(data);
	data = NULL;
#endif

	// Read config block
	WMR_INFO(wcb, "Reading %s controller config",
	         wcb->base.device_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER ? "left" : "right");

	// Check if we have it cached already
	char *cache_filename = build_cache_filename(serial_no);

	if (!read_calibration_cache(wcb, cache_filename)) {
		unsigned char *data = NULL;
		size_t data_size;

		ret = wmr_read_fw_block(wcb, 0x02, &data, &data_size);
		if (ret < 0 || data == NULL || data_size < 2) {
			free(cache_filename);
			return false;
		}

		/* De-obfuscate the JSON config */
		config_json_block = data + sizeof(uint16_t);
		for (unsigned int i = 0; i < data_size - sizeof(uint16_t); i++) {
			config_json_block[i] ^= wmr_config_key[i % sizeof(wmr_config_key)];
		}

		if (!wmr_controller_config_parse(&wcb->config, (char *)config_json_block, wcb->log_level)) {
			free(cache_filename);
			free(data);
			return false;
		}

		/* Write to the cache file (if it fails, ignore it, it's just a cache) */
		write_calibration_cache(wcb, cache_filename, config_json_block, data_size - sizeof(uint16_t));
		free(data);
	} else {
		WMR_DEBUG(wcb, "Read %s controller config from cache %s",
		          wcb->base.device_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER ? "left" : "right",
		          cache_filename);
	}
	free(cache_filename);

	// Seed the IMU fusion with this unit's persisted calibration prior (created at line 677, so it
	// exists by now). Keep the serial for the matching save at deinit.
	strncpy(wcb->imu_cal_serial, serial_no, sizeof(wcb->imu_cal_serial) - 1);
	wcb->imu_cal_serial[sizeof(wcb->imu_cal_serial) - 1] = '\0';
	imu_cal_load(wcb, wcb->imu_cal_serial);

	WMR_DEBUG(wcb, "Parsed %d LED entries from controller calibration", wcb->config.led_count);
	return true;
}

/* Live HMD pose in the controller's world frame (OpenXR), at @p at_timestamp_ns. Returns NULL if no HMD is
 * attached or its pose is unavailable, so the fusion degrades to world-frame holds (never a bad reference).
 * XRT_INPUT_GENERIC_TRACKER_POSE is the raw head pose the constellation tracker uses as its camera base —
 * the SAME world frame the controller poses live in — so controller-minus-HMD is a valid head-relative offset. */
static const struct xrt_pose *
wmr_controller_query_hmd_pose(struct wmr_controller_base *wcb, int64_t at_timestamp_ns, struct xrt_pose *out_pose)
{
	if (wcb->hmd_xdev == NULL) {
		return NULL;
	}
	struct xrt_space_relation hmd_rel = {0};
	if (xrt_device_get_tracked_pose(wcb->hmd_xdev, XRT_INPUT_GENERIC_TRACKER_POSE, at_timestamp_ns, &hmd_rel) !=
	    XRT_SUCCESS) {
		return NULL;
	}
	const enum xrt_space_relation_flags need =
	    XRT_SPACE_RELATION_POSITION_VALID_BIT | XRT_SPACE_RELATION_ORIENTATION_VALID_BIT;
	if ((hmd_rel.relation_flags & need) != need) {
		return NULL; // head pose not yet valid (SLAM not converged): no reference
	}
	*out_pose = hmd_rel.pose;
	return out_pose;
}

static xrt_result_t
wmr_controller_base_get_pose_pair(struct xrt_device *xdev,
                                     enum xrt_input_name name,
                                     int64_t at_timestamp_ns,
                                     struct xrt_space_relation *out_relation,
                                     struct xrt_pose *out_from_raw, uint64_t *out_generation,
                                     const struct xrt_pose *presented_hmd_in_tracking)
{
	DRV_TRACE_MARKER();

	struct wmr_controller_base *wcb = wmr_controller_base(xdev);

	struct xrt_relation_chain xrc = {0};
	struct xrt_space_relation relation = {0};

	m_relation_chain_push_pose(&xrc, &wcb->P_aim);
	if (name == XRT_INPUT_G2_CONTROLLER_GRIP_POSE || name == XRT_INPUT_ODYSSEY_CONTROLLER_GRIP_POSE ||
	    name == XRT_INPUT_WMR_GRIP_POSE) {
		m_relation_chain_push_pose(&xrc, &wcb->P_aim_grip);
	}

	// Position + orientation both come from the tightly-coupled per-LED fusion (the ESKF). Hand it the live
	// HMD pose so an out-of-view controller body-locks (rides with the head at arm's reach) instead of
	// dead-reckoning to metres in world frame.
	struct xrt_pose hmd_pose;
	if (out_from_raw != NULL) {
		// OpenVR's paired head pose is middle-eye. Body/reach logic historically uses the IMU
		// reference: remove only its fixed local extrinsic, without another raw head query.
		const struct xrt_pose *presented_imu = presented_hmd_in_tracking;
		if (presented_hmd_in_tracking != NULL && wcb->hmd_xdev != NULL) {
			struct wmr_hmd *hmd = wmr_hmd(wcb->hmd_xdev);
			if (hmd->tracking.slam_enabled && hmd->slam_over_3dof && hmd->tracking.imu2me) {
				struct xrt_pose me_to_imu;
				math_pose_invert(&hmd->config.sensors.transforms.P_imu_me, &me_to_imu);
				math_pose_transform(presented_hmd_in_tracking, &me_to_imu, &hmd_pose);
				presented_imu = &hmd_pose;
			}
		}
		kalman_fusion_get_prediction_with_presentation(wcb->kalman_fusion, at_timestamp_ns, &relation,
		                                              presented_imu, out_from_raw, out_generation);
	} else {
		const struct xrt_pose *hmd_world_pose = wmr_controller_query_hmd_pose(wcb, at_timestamp_ns, &hmd_pose);
		kalman_fusion_get_prediction(wcb->kalman_fusion, at_timestamp_ns, &relation, hmd_world_pose);
	}

	m_relation_chain_push_relation(&xrc, &relation);
	m_relation_chain_resolve(&xrc, out_relation);

	wcb->pose = out_relation->pose;
	return XRT_SUCCESS;
}

static xrt_result_t
wmr_controller_base_get_tracked_pose(struct xrt_device *xdev, enum xrt_input_name name,
                                    int64_t at_timestamp_ns, struct xrt_space_relation *out_relation)
{
    return wmr_controller_base_get_pose_pair(xdev, name, at_timestamp_ns, out_relation, NULL, NULL, NULL);
}

void
wmr_controller_base_deinit(struct wmr_controller_base *wcb)
{
	DRV_TRACE_MARKER();

	// Remove the variable tracking.
	u_var_remove_root(wcb);

	// Disconnect from the connection so we don't
	// receive any more callbacks
	os_mutex_lock(&wcb->conn_lock);
	struct wmr_controller_connection *conn = wcb->wcc;
	wcb->wcc = NULL;
	os_mutex_unlock(&wcb->conn_lock);

	if (conn != NULL) {
		wmr_controller_connection_disconnect(conn);
	}

	if (wcb->tracking_connection) {
		t_constellation_tracked_device_connection_disconnect(wcb->tracking_connection);
		wcb->tracking_connection = NULL;
	}

	m_clock_windowed_skew_tracker_destroy(wcb->hw2mono_clock);

	os_mutex_destroy(&wcb->conn_lock);
	os_mutex_destroy(&wcb->data_lock);

	if (wcb->kalman_fusion) {
		imu_cal_save(wcb, wcb->imu_cal_serial); // persist this unit's converged IMU calibration (EMA)
		kalman_fusion_destroy(wcb->kalman_fusion);
	}
}

/*
 *
 * 'Exported' functions.
 *
 */

bool
wmr_controller_base_init(struct wmr_controller_base *wcb,
                         struct wmr_controller_connection *conn,
                         enum xrt_device_type controller_type,
                         enum u_logging_level log_level,
                         u_device_destroy_function_t destroy_fn)
{
	DRV_TRACE_MARKER();

	wcb->log_level = log_level;
	wcb->wcc = conn;
	wcb->receive_bytes = receive_bytes;
	wcb->pose = (struct xrt_pose)XRT_POSE_IDENTITY;

	// IMU samples arrive every 5ms on average
	// 1 second seems to be enough to smooth things
	const int IMU_ARRIVAL_FREQ = 200;
	wcb->hw2mono_clock = m_clock_windowed_skew_tracker_alloc(IMU_ARRIVAL_FREQ);

	// Constant controller-IMU vs headset-camera clock offset (ns), added to optical timestamps before
	// fusion. The OOSM handles the variable processing lag; this is the residual fixed sensor-pair
	// skew, measured at +4.8 ms (see wmr_g2_timing.h); recalibratable via G2_CTRL_TD_NS.
	const char *td_env = getenv("G2_CTRL_TD_NS");
	wcb->ctrl_optical_td_ns = (td_env != NULL) ? (int64_t)atoll(td_env) : WMR_CTRL_OPTICAL_TD_DEFAULT_NS;

	if (controller_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER) {
		snprintf(wcb->base.str, ARRAY_SIZE(wcb->base.str), "WMR Left Controller");
		/* TODO: use proper serial from read_controller_config()? */
		snprintf(wcb->base.serial, XRT_DEVICE_NAME_LEN, "Left Controller");
	} else {
		snprintf(wcb->base.str, ARRAY_SIZE(wcb->base.str), "WMR Right Controller");
		/* TODO: use proper serial from read_controller_config()? */
		snprintf(wcb->base.serial, XRT_DEVICE_NAME_LEN, "Right Controller");
	}

	// Set all functions.
	u_device_populate_function_pointers(&wcb->base, wmr_controller_base_get_tracked_pose, destroy_fn);
	wcb->base.get_tracked_pose_with_presentation = wmr_controller_base_get_pose_pair;

	wcb->base.name = XRT_DEVICE_WMR_CONTROLLER;
	wcb->base.device_type = controller_type;
	wcb->base.supported.orientation_tracking = true;
	wcb->base.supported.position_tracking = true;
	wcb->base.supported.hand_tracking = false;

	/* Default grip pose up by 35° degrees around the X axis and
	 * back about 10cm (back is +Z in OXR coords), but overridden
	 * by subclasses with real values from controller models */
	struct xrt_vec3 translation = {0.0, 0, 0.1};
	struct xrt_vec3 axis = {1.0, 0, 0};
	math_quat_from_angle_vector(DEG_TO_RAD(35), &axis, &wcb->P_aim_grip.orientation);
	wcb->P_aim_grip.position = translation;

	struct xrt_vec3 aim_translation = {-0.014322, 0.018838, 0};
	wcb->P_aim.position = aim_translation;
	struct xrt_vec3 aim_axis = {0.0, 1.0, 0};
	math_quat_from_angle_vector(DEG_TO_RAD(12.5), &aim_axis, &wcb->P_aim.orientation);

	wcb->thumbstick_deadzone = 0.15;

	wcb->kalman_fusion = kalman_fusion_create();

	if (os_mutex_init(&wcb->conn_lock) != 0 || os_mutex_init(&wcb->data_lock) != 0) {
		WMR_ERROR(wcb, "WMR Controller: Failed to init mutex!");
		return false;
	}

	/* Send init commands */
	struct wmr_controller_fw_cmd fw_cmd = {
	    0,
	};
	struct wmr_controller_fw_cmd_response fw_cmd_response;

	/* Zero command. Reinits controller internal state */
	fw_cmd = WMR_CONTROLLER_FW_CMD_INIT(0x06, 0x0, 0, 0);
	if (wmr_controller_send_fw_cmd(wcb, &fw_cmd, 0x06, &fw_cmd_response) < 0) {
		return false;
	}

	/* Quiesce/restart controller tasks */
	fw_cmd = WMR_CONTROLLER_FW_CMD_INIT(0x06, 0x04, 0xc1, 0x02);
	if (wmr_controller_send_fw_cmd(wcb, &fw_cmd, 0x06, &fw_cmd_response) < 0) {
		return false;
	}

	// Read config file from controller
	if (!read_controller_config(wcb)) {
		return false;
	}

	wmr_config_precompute_transforms(&wcb->config.sensors, NULL);

	// Announce the availabilty of the config for the tracker to retrieve
	os_mutex_lock(&wcb->data_lock);
	wcb->have_config = true;
	os_mutex_unlock(&wcb->data_lock);

	/* Reset device time before controller outputs */
	fw_cmd = WMR_CONTROLLER_FW_CMD_INIT(0x06, 0x21, 0x00, 0x00);
	if (wmr_controller_send_fw_cmd(wcb, &fw_cmd, 0x06, &fw_cmd_response) < 0) {
		return false;
	}

	/* Enable the status reports, IMU and control status reports */
	const unsigned char wmr_controller_status_enable_cmd[64] = {0x06, 0x03, 0x01, 0x00, 0x02};
	wmr_controller_send_bytes(wcb, wmr_controller_status_enable_cmd, sizeof(wmr_controller_status_enable_cmd));
	os_nanosleep(U_TIME_1MS_IN_NS * 20); // Sleep 20ms
	const unsigned char wmr_controller_imu_on_cmd[64] = {0x06, 0x03, 0x02, 0xe1, 0x02};
	wmr_controller_send_bytes(wcb, wmr_controller_imu_on_cmd, sizeof(wmr_controller_imu_on_cmd));

	wcb->timesync_counter = 2;
	/* Start at full drive so acquisition sees the brightest possible LEDs; the brightness control
	 * loop trims it down only if a close controller reads above LED_BRIGHT_TARGET_HI (bloom regime). */
	wcb->timesync_led_intensity = 399;
	wcb->timesync_val2 = 500;
	wcb->timesync_time_offset = 0;

	wcb->timesync_led_intensity_uvar =
	    (struct u_var_draggable_u16){.val = &wcb->timesync_led_intensity, .min = 1, .max = 399, .step = 1};
	wcb->timesync_val2_uvar =
	    (struct u_var_draggable_u16){.val = &wcb->timesync_val2, .min = 0, .max = 1023, .step = 1};
	wcb->timesync_time_offset_uvar =
	    (struct u_var_draggable_u16){.val = &wcb->timesync_time_offset, .min = 0, .max = 44, .step = 1};

	u_var_add_root(wcb, wcb->base.str, true);
	u_var_add_log_level(wcb, &wcb->log_level, "Log Level");
	u_var_add_pose(wcb, &wcb->pose, "Reported pose");

	u_var_add_gui_header(wcb, NULL, "IMU");
	u_var_add_ro_vec3_f32(wcb, &wcb->last_imu.acc, "imu.accel");
	u_var_add_ro_vec3_f32(wcb, &wcb->last_imu.gyro, "imu.gyro");
	u_var_add_i32(wcb, &wcb->last_imu.temperature, "imu.temperature");
	u_var_add_ro_u64(wcb, &wcb->last_imu_timestamp_ns, "Last CPU IMU TS");
	u_var_add_ro_u64(wcb, &wcb->last_imu_device_timestamp_ns, "Last device IMU TS");

	u_var_add_gui_header(wcb, NULL, "Optical Tracking");
	u_var_add_pose(wcb, &wcb->last_tracked_pose, "Last observed pose");
	u_var_add_ro_i64(wcb, &wcb->last_tracked_pose_ts, "Last observed pose TS");
	u_var_add_ro_u16(wcb, &wcb->last_brightness_report, "Last observed average LED brightness");

	u_var_add_gui_header(wcb, NULL, "Kalman Fusion");
	kalman_fusion_add_ui(wcb->kalman_fusion, wcb,
	                     (wcb->base.device_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER) ? "wmr_left"
	                                                                                     : "wmr_right");

	u_var_add_gui_header(wcb, NULL, "LED Sync");
	u_var_add_draggable_u16(wcb, &wcb->timesync_led_intensity_uvar, "LED intensity");
	u_var_add_draggable_u16(wcb, &wcb->timesync_val2_uvar, "U2");
	u_var_add_draggable_u16(wcb, &wcb->timesync_time_offset_uvar, "time offset (0.5ms increment)");
	u_var_add_ro_u64(wcb, &wcb->last_timesync_timestamp_ns, "Last CPU timesync TS");
	u_var_add_ro_u64(wcb, &wcb->last_timesync_device_timestamp_ns, "Last device timesync TS");

	u_var_add_gui_header(wcb, NULL, "Misc");
	u_var_add_pose(wcb, &wcb->P_aim, "Aim pose offset");
	u_var_add_pose(wcb, &wcb->P_aim_grip, "Grip pose offset");
	u_var_add_ro_u64(wcb, &wcb->next_keepalive_timestamp_ns, "Next keepalive TS");

	return true;
}

/*
 * Timesync packet format:
 *  XX    YY    AA    BB    CC    CC    CC    CC    CC    CC    DD    EE
 *   0     1     2     3     4     5     6     7     8     9    10    11
 *
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |0              |    1          |        2      |            3  |
 * |0 1 2 3 4 5 6 7|8 9 0 1 2 3 4 5|6 7 8 9 0 1 2 3|4 5 6 7 8 9 0 1|
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |       XX      |      YY       | C |  U1[0:5]  :[6:8]| TS[0:4] |  XX YY AA BB
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                           TS[5:36]                            |  CC CC CC CC
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |         TS[37:54]                 |   U2[0:10]          | F2  |  CC CC DD EE
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *
 * XX = 0x3
 * YY = 8 bit counter that increments sequentially across timesync and keepalives
 * C  = 2 bit counter. Always starts as 1, then counts 1,2,3 per packet,
 *      regardless of the time between. Must not be 0
 * U1 = Led intensity / pulse length
 *       - clamped at 1..399 in setLEDPulseLengthMaybe()
 * TS = 55-bit sync timestamp (in µS) taken from camera exposure timings
 * U2 = unknown 11-bits
 *      - Gets multiplied by 2 and has 1000 added before use?
 *      - Looks like it gets extended to 64-bit and used in an unclear way
 *      - Looks like it should affect something in the HID status report packet
 *      - First WMR packet is always 800
 *      - Minimum value seen from WMR is 0, max (after the initial 800) is 1023
 * F  = flags? 3 bits. Allowed values seem to be 1,2,3,4
 *      - WMR always sends 1. Seems like 2 might be 'off'
 *      - Referred to as 'LED train type'
 */
static void
fill_timesync_packet(
    uint8_t buf[12], uint8_t cmd_ctr, uint8_t ts_ctr, int led_intensity, uint64_t ts, int U2, uint8_t flags)
{
	ts_ctr = (ts_ctr & 0x3);
	led_intensity = CLAMP(led_intensity, 1, 399);
	U2 = CLAMP(U2, 0, 1023);

	buf[0] = 0x3;
	buf[1] = cmd_ctr;
	buf[2] = ts_ctr | ((led_intensity & 0x3f) << 2);
	buf[3] = ((led_intensity >> 6) & 0x7) | ((ts & 0x1f) << 3);
	buf[4] = ts >> 5;
	buf[5] = ts >> 13;
	buf[6] = ts >> 21;
	buf[7] = ts >> 29;
	buf[8] = ts >> 37;
	buf[9] = ts >> 45;
	buf[10] = ((ts >> 53) & 0x3) | (U2 << 2);
	buf[11] = ((U2 >> 6) & 0x1f) | ((flags & 0x3) << 5);
}

/* Called with data_lock held */
static void
wmr_controller_base_send_timesync(struct wmr_controller_base *wcb)
{
	/* @todo: Check if timesync values have changed and skip sending if not */
	uint8_t timesync_pkt[12];

	os_mutex_lock(&wcb->conn_lock);
	struct wmr_controller_connection *conn = wcb->wcc;
	if (conn != NULL) {
		/* Each timesync_time_offset step is 0.5ms */
		uint64_t time_offset_us = wcb->timesync_time_offset * 500;

		/* Timesync counter counts 1/2/3 in a loop per packet */
		uint8_t ts_ctr = wcb->timesync_counter++;
		if (wcb->timesync_counter == 4) {
			wcb->timesync_counter = 1;
		}

		int led_intensity = wcb->timesync_led_intensity;
		uint64_t slam_time_us = wcb->timesync_device_slam_time_us + time_offset_us;
		int val2 = wcb->timesync_val2;

		fill_timesync_packet(timesync_pkt, wcb->cmd_counter++, ts_ctr, led_intensity, slam_time_us, val2, 1);
		os_mutex_unlock(&wcb->data_lock);
		wmr_controller_connection_send_bytes(conn, timesync_pkt, sizeof(timesync_pkt));

		os_mutex_unlock(&wcb->conn_lock);
		os_mutex_lock(&wcb->data_lock);

		WMR_DEBUG(
		    wcb, "%s controller timesync counter %u led_intensity %u time %" PRIu64 " offset %" PRIu64 " U2 %u",
		    wcb->base.device_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER ? "Left" : "Right", ts_ctr,
		    wcb->timesync_led_intensity, slam_time_us, time_offset_us, val2);
		WMR_TRACE_HEX(wcb, timesync_pkt, sizeof(timesync_pkt));
	} else {
		os_mutex_unlock(&wcb->conn_lock);
	}
}

static void
wmr_controller_base_notify_frame(struct xrt_device *xdev, uint64_t frame_mono_ns, uint64_t frame_sequence)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)xdev;

	os_mutex_lock(&wcb->data_lock);

	/* The frame cadence is SLAM/controller/controller. Only controller frames are received
	 * here. On the 2nd controller frame, we need to pass the estimate of the time of the next
	 * *1st controller frame* minus 1/3 of a frame duration to the controller timesync methods.
	 * That is 2/90 + 1/270 = 5/270 or 5/3rds of the average frame duration.
	 * @todo: Also calculate how many full 3-frame cycles have passed since the last
	 * '2nd Controller frame' to add to our estimate in case of scheduling delays.
	 * @todo: Move the 'next frame' estimate into the controller timesync code so it's
	 * calculated when the led control packet is actually about to be sent.
	 * Since only controller frames get passed to here, the 2nd sequential controller
	 * frame is the one right before the next SLAM frame */
	bool is_second_frame = (wcb->last_frame_sequence + 1) == frame_sequence;

	if (is_second_frame) {
		// on the 2nd sequential (controller) frame, update the controller timesync estimate of the next SLAM
		// frame's time
		timepoint_ns next_slam_mono_ns = (timepoint_ns)(frame_mono_ns) + U_TIME_1MS_IN_NS * 18;

		timepoint_ns next_device_slam_time_ns;
		if (m_clock_windowed_skew_tracker_to_remote(wcb->hw2mono_clock, next_slam_mono_ns,
		                                            &next_device_slam_time_ns)) {
			uint64_t next_device_slam_time_us = next_device_slam_time_ns / 1000;

			if (next_device_slam_time_us != wcb->timesync_device_slam_time_us) {
				wcb->timesync_device_slam_time_us = next_device_slam_time_us;
				wcb->timesync_updated = true;
			}

			wcb->last_timesync_device_timestamp_ns = next_device_slam_time_ns;
			wcb->last_timesync_timestamp_ns = next_slam_mono_ns;
		}
	}

	wcb->last_frame_timestamp = frame_mono_ns;
	wcb->last_frame_sequence = frame_sequence;

	os_mutex_unlock(&wcb->data_lock);
}

/* Called with data_lock held */
static void
wmr_controller_base_send_keepalive(struct wmr_controller_base *wcb, uint64_t now_ns)
{
	const uint64_t KEEPALIVE_DURATION = 125 * U_TIME_1MS_IN_NS;

	if (wcb->next_keepalive_timestamp_ns > now_ns) {
		/* Too soon to send the Keepalive */
		return;
	}


	os_mutex_lock(&wcb->conn_lock);
	struct wmr_controller_connection *conn = wcb->wcc;
	if (conn != NULL) {
		uint8_t keepalive_pkt[2];

		keepalive_pkt[0] = WMR_MOTION_CONTROLLER_KEEPALIVE;
		keepalive_pkt[1] = wcb->cmd_counter++;

		os_mutex_unlock(&wcb->data_lock);
		wmr_controller_connection_send_bytes(conn, keepalive_pkt, sizeof(keepalive_pkt));

		os_mutex_unlock(&wcb->conn_lock);
		os_mutex_lock(&wcb->data_lock);
	}

	/* Calculate the next timeout */
	if (wcb->next_keepalive_timestamp_ns == 0) {
		wcb->next_keepalive_timestamp_ns = now_ns + KEEPALIVE_DURATION;
	} else {
		wcb->next_keepalive_timestamp_ns += KEEPALIVE_DURATION;
	}
}

#define WMR_RING_HEIGHT 0.02194146618190565
#define WMR_RING_TOP_RADIUS (0.11277887330599087 / 2.0)
#define WMR_RING_BOTTOM_RADIUS (0.09375531956362483 / 2.0)

static bool
conical_frustum_ray_intersect(struct xrt_vec3 ray_origin,
                              struct xrt_vec3 ray_dir,
                              struct xrt_vec3 base_center,
                              struct xrt_vec3 axis,
                              float h,
                              float r1,
                              float r2,
                              float *hit_time)
{
	// cone has it's base on the bottom, must shrink as it goes up
	assert(r1 > r2);

	// compute cone slope k = (r1 - r2) / h
	float k = (r1 - r2) / h;
	float k2 = k * k;

	// apex of the cone
	float cone_height = r1 / k;
	struct xrt_vec3 apex = m_vec3_sub(base_center, m_vec3_mul_scalar(axis, cone_height));

	// vector from apex to ray origin
	struct xrt_vec3 delta_p = m_vec3_sub(ray_origin, apex);

	// dot products
	float dv = m_vec3_dot(ray_dir, axis);
	float pv = m_vec3_dot(delta_p, axis);

	// quadratic coefficients
	float a = m_vec3_dot(ray_dir, ray_dir) - (1 + k2) * dv * dv;
	float b = 2 * (m_vec3_dot(ray_dir, delta_p) - (1 + k2) * dv * pv);
	float c = m_vec3_dot(delta_p, delta_p) - (1 + k2) * pv * pv;

	// discriminant
	float discriminant = b * b - 4 * a * c;
	if (discriminant < -1e-6f) // allow some small numerical tolerance
		return false;

	if (discriminant < 0.0f)
		discriminant = 0.0f;

	float sqrt_d = sqrtf(discriminant);
	float t0 = (-b - sqrt_d) / (2 * a);
	float t1 = (-b + sqrt_d) / (2 * a);

	// pick nearest positive intersection
	float t = t0 > 0 ? t0 : t1;
	if (t < 0)
		return false;

	// intersection point
	struct xrt_vec3 p = m_vec3_add(ray_origin, m_vec3_mul_scalar(ray_dir, t));

	// project onto cone axis to check vertical bounds
	float u = m_vec3_dot(m_vec3_sub(p, base_center), axis);
	if (u < 0 || u > h)
		return false;

	if (hit_time)
		*hit_time = t;

	return true;
}

static bool
wmr_controller_base_check_led_visibility(struct t_constellation_led_model *led_model,
                                         size_t led_index,
                                         struct xrt_vec3 T_obj_cam)
{
	// @todo *so much* of this can be pre-computed... but this is fine for now.

	struct t_constellation_led *led = &led_model->leds[led_index];

	struct xrt_vec3 led_dir = led->dir;
	led_dir.z = 0;
	math_vec3_normalize(&led_dir);

	struct xrt_vec3 led_dir_to_z_axis = m_vec3_inverse((struct xrt_vec3){led->pos.x, led->pos.y, 0});
	math_vec3_normalize(&led_dir_to_z_axis);

	float angle_away_from_origin = fabsf(acosf(m_vec3_dot(led_dir_to_z_axis, led_dir)));

	struct xrt_vec3 ring_base_pos = {0, 0, (WMR_RING_HEIGHT / 2.0)};
	struct xrt_vec3 ring_base_rot = {0, 0, -1};

	// for inward-facing LEDs, check if they intersect with the Cone
	if (angle_away_from_origin < DEG_TO_RAD(30.0) &&
	    conical_frustum_ray_intersect(led->pos, m_vec3_normalize(m_vec3_sub(T_obj_cam, led->pos)), ring_base_pos,
	                                  ring_base_rot, WMR_RING_HEIGHT, WMR_RING_TOP_RADIUS, WMR_RING_BOTTOM_RADIUS,
	                                  NULL)) {
		return false;
	}

	return true;
}

static bool
wmr_controller_base_get_led_model(struct xrt_device *xdev, struct t_constellation_led_model *led_model)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);

	os_mutex_lock(&wcb->data_lock);
	if (!wcb->have_config) {
		os_mutex_unlock(&wcb->data_lock);
		return false;
	}
	os_mutex_unlock(&wcb->data_lock);

	t_constellation_led_model_init((int)wcb->base.device_type, NULL, led_model, wcb->config.led_count, 8);
	led_model->check_led_visibility = wmr_controller_base_check_led_visibility;

	// Note: This LED model is in OpenCV/WMR coordinates with
	// XYZ = Right/Down/Forward
	for (int i = 0; i < wcb->config.led_count; i++) {
		struct t_constellation_led *led = led_model->leds + i;

		led->id = i;

		struct wmr_led_config *wmr_led = &wcb->config.leds[i];
		led->pos = wmr_led->pos;
		led->dir = wmr_led->norm;

		led->radius_mm = 3;
	}

	const float controller_height = 0.150; // controller is about 150mm high
	const float bounding_box_back = -(controller_height - (WMR_RING_HEIGHT / 2.0));

	const float base_radius = 0.070 / 2.0; // the width between the the left and right of the controller's face

	struct xrt_vec3 bounding_points[8] = {
	    {WMR_RING_TOP_RADIUS, WMR_RING_TOP_RADIUS, WMR_RING_HEIGHT / 2},   // +Z, +X, +Y
	    {-WMR_RING_TOP_RADIUS, WMR_RING_TOP_RADIUS, WMR_RING_HEIGHT / 2},  // +Z, -X, +Y
	    {WMR_RING_TOP_RADIUS, -WMR_RING_TOP_RADIUS, WMR_RING_HEIGHT / 2},  // +Z, +X, -Y
	    {-WMR_RING_TOP_RADIUS, -WMR_RING_TOP_RADIUS, WMR_RING_HEIGHT / 2}, // +Z, -X, -Y
	    {base_radius, base_radius, bounding_box_back},                     // -Z, +X, +Y
	    {-base_radius, base_radius, bounding_box_back},                    // -Z, -X, +Y
	    {base_radius, -base_radius, bounding_box_back},                    // -Z, +X, -Y
	    {-base_radius, -base_radius, bounding_box_back},                   // -Z, -X, -Y
	};

	for (size_t i = 0; i < ARRAY_SIZE(bounding_points); i++) {
		led_model->bounding_points[i].pos = bounding_points[i];
	}

	t_constellation_led_model_dump(led_model, wcb->base.str);

	return true;
}

static bool
wmr_controller_base_get_estimator_prior(struct xrt_device *xdev, timepoint_ns when_ns,
                                      struct t_estimator_prior *out)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)xdev;
	if (!wcb->kalman_fusion) { return false; }
	kalman_fusion_get_estimator_prior(wcb->kalman_fusion, when_ns + wcb->ctrl_optical_td_ns, out);
	return true; // supported even without usable history: cold optical bootstrap remains possible
}

static bool
wmr_controller_base_validate_prior_epoch(struct xrt_device *xdev, const struct t_estimator_prior *prior)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)xdev;
	return wcb->kalman_fusion && prior &&
	       kalman_fusion_get_world_generation(wcb->kalman_fusion) == prior->world_generation;
}

static bool
wmr_controller_base_predict_led_gate_from_prior(const struct t_estimator_prior *prior,
    const struct xrt_pose *P_xrworld_cam, const struct t_constellation_cam_calib *calib,
    const struct xrt_vec3 *led_obj, float out_zhat[2], float out_S[4])
{
	struct kalman_led_camera_view view = {
	    .fx = calib->fx, .fy = calib->fy, .cx = calib->cx, .cy = calib->cy,
	    .cam_world_orient = P_xrworld_cam->orientation, .cam_world_pos = P_xrworld_cam->position,
	};
	struct kalman_led_observation obs = {.led_obj = *led_obj};
	return kalman_fusion_predict_led_gate_from_prior(prior, &obs, &view, out_zhat, out_S);
}

static bool
wmr_controller_base_get_pose_uncertainty(struct xrt_device *xdev,
                                         double *position_std,
                                         double *orientation_std,
                                         double *yaw_std,
                                         double *tilt_std)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	if (wcb->kalman_fusion == NULL) {
		return false;
	}
	// The fusion reads its published snapshot wait-free, so no data_lock is needed here.
	return kalman_fusion_get_pose_uncertainty(wcb->kalman_fusion, position_std, orientation_std, yaw_std,
	                                          tilt_std);
}

static bool
wmr_controller_base_get_gravity_tilt_reference(struct xrt_device *xdev,
                                               struct xrt_quat *out_gravity_corrected_q,
                                               double *out_excess_m_s2)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	if (wcb->kalman_fusion == NULL) {
		return false;
	}
	return kalman_fusion_get_gravity_tilt_reference(wcb->kalman_fusion, out_gravity_corrected_q,
	                                                out_excess_m_s2);
}

static bool
wmr_controller_base_get_predicted_pose(struct xrt_device *xdev,
                                       timepoint_ns when_ns,
                                       struct xrt_space_relation *out_relation)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	if (wcb->kalman_fusion == NULL) {
		return false;
	}
	// Compatibility entry point; the production matcher acquires the complete historical bundle once.
	kalman_fusion_get_predicted_pose(wcb->kalman_fusion, when_ns + wcb->ctrl_optical_td_ns, out_relation);
	return true;
}

static bool
wmr_controller_base_get_last_optical_age_ms(struct xrt_device *xdev, timepoint_ns when_ns, double *age_ms)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	if (wcb->kalman_fusion == NULL || age_ms == NULL) {
		return false;
	}
	return kalman_fusion_debug_get_last_optical_age_ms(wcb->kalman_fusion, when_ns + wcb->ctrl_optical_td_ns,
	                                                   age_ms);
}

/* Observed LED peak-brightness band (0-255 image counts) the LED-drive control loop rides toward.
 * Was 30-70 -- far too dim: detected LEDs peaked at a median of ~54, barely above the blob admission
 * cut (pixel floor 8 + local background + 6 margin), so dim/angled/far LEDs fell below it and only ~5
 * of ~13 visible LEDs were matched -> degenerate few-point PnP -> mirror flips + fly-off. Drive the
 * LEDs as bright as the detector can actually use instead: most LEDs then clear the cut and the
 * sub-pixel centroids stay crisp. Saturated cores are fine (blobwatch centroids them edge-symmetrically);
 * the only regime this band backs off from is gross bloom/merge of adjacent LEDs at very close range.
 * The band is denominated in image DN on purpose: its remaining job is anti-bloom backoff near the 255
 * ADC clip, which does not move with camera gain (B3 design §1c) -- do not scale it. */
#define LED_BRIGHT_TARGET_LO 160
#define LED_BRIGHT_TARGET_HI 200

static void
wmr_controller_base_push_brightness_update(struct xrt_device *xdev, uint8_t average_brightness)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	os_mutex_lock(&wcb->data_lock);

	wcb->last_brightness_report = average_brightness;

	if (average_brightness > LED_BRIGHT_TARGET_HI) {
		wcb->timesync_led_intensity -= MIN(wcb->timesync_led_intensity, 3);
	}

	if (average_brightness < LED_BRIGHT_TARGET_LO) {
		wcb->timesync_led_intensity = MIN(wcb->timesync_led_intensity + 10, 399);
	}

	os_mutex_unlock(&wcb->data_lock);
}

// Per-LED gate, in camera PIXELS (the obs are undistorted pixels under the view's real intrinsics, so it
// is focal-independent). The blob-centroid measurement noise R is owned by the filter (= LED_PIXEL_STD²);
// we pass NULL variance so there is a SINGLE source of truth (passing a separate value here is what made
// the live fold use std-as-variance). MAX_INNOV_PX rejects a mislabelled LED whose reprojection is farther
// than this from the filter's predicted pose.
#define ESKF_MAX_INNOV_PX 8.0f // px

// Tightly-coupled per-LED optical feed (the ESKF). Translates the constellation's neutral per-view
// payload into the fusion's structs and folds each LED. The blobs are undistorted to pixels and the
// view carries the camera's REAL pinhole intrinsics, so the measurement noise + gate are in physical
// pixels (focal-independent); the extrinsic carries the OpenXR<->OpenCV YZ flip.
static void
wmr_controller_base_push_observed_leds(struct xrt_device *xdev,
                                       timepoint_ns frame_mono_ns,
                                       const struct xrt_pose *P_xrworld_cam,
                                       const struct t_constellation_cam_calib *cam_calib,
                                       const struct t_constellation_led_obs *leds,
                                       size_t led_count)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	if (wcb->kalman_fusion == NULL || P_xrworld_cam == NULL || cam_calib == NULL || leds == NULL ||
	    led_count == 0) {
		return;
	}

	// Real pinhole intrinsics for this view (the obs are undistorted pixels under exactly these).
	struct kalman_led_camera_view view = {
	    .fx = cam_calib->fx,
	    .fy = cam_calib->fy,
	    .cx = cam_calib->cx,
	    .cy = cam_calib->cy,
	    .cam_world_orient = P_xrworld_cam->orientation,
	    .cam_world_pos = P_xrworld_cam->position,
	};

	// Copy into the fusion's struct (identical layout, different field names). 64 == the
	// constellation's MAX_OBJECT_LEDS upper bound; clamp defensively.
	struct kalman_led_observation obs[64];
	size_t n = led_count < 64 ? led_count : 64;
	for (size_t i = 0; i < n; i++) {
		obs[i].observed_px = leds[i].obs_px;
		obs[i].led_obj = leds[i].led_obj;
		obs[i].pos_var_px2 = leds[i].pos_var_px2;
	}

	// Live HMD pose at this frame's capture time: the body-lock reference + arm-reach gate origin. Queried
	// at frame_mono_ns (the LED capture time, the same instant the constellation places the cameras).
	struct xrt_pose hmd_pose;
	const struct xrt_pose *hmd_world_pose = wmr_controller_query_hmd_pose(wcb, frame_mono_ns, &hmd_pose);

	os_mutex_lock(&wcb->data_lock);
	// Tightly-coupled per-LED fold (primary optical path); the gate skips a mislabelled LED whose
	// reprojection is far from the filter's predicted pose. Returns LEDs folded (-1 if awaiting bootstrap).
	// NULL variance => the filter's own LED_PIXEL_STD² (single source of truth for R).
	// td: align the optical capture time onto the IMU clock before fusion (default 0).
	float folded = kalman_fusion_process_led_observations(wcb->kalman_fusion,
	                                                      frame_mono_ns + wcb->ctrl_optical_td_ns, obs, n,
	                                                      &view, NULL, ESKF_MAX_INNOV_PX, /*feed=*/true,
	                                                      hmd_world_pose);
	os_mutex_unlock(&wcb->data_lock);

		if (g2_telem_enabled() && folded >= 0.0f) {
			uint8_t id = wmr_controller_telem_id(wcb);
			g2_telem_event(id, (uint64_t)frame_mono_ns, G2_TELEM_EV_ESKF_FOLD_COUNT, folded);
			g2_telem_event(id, (uint64_t)frame_mono_ns, G2_TELEM_EV_ESKF_LEDS_SEEN, (float)n);
		}
}

// Per-LED ANISOTROPIC covariance gate: predict one LED's image point + 2x2 innovation covariance S =
// H·P·Hᵀ + R from the filter's LIVE covariance (same projection/Jacobian as the fold). Used by the
// front-end covariance-driven associator to gate a candidate blob<->LED pairing by its Mahalanobis
// distance. Returns false until the filter is tracking (no usable prior) — the front-end then does NOT
// gate-fold (cold-start guard). The payload mirrors wmr_controller_base_push_observed_leds exactly.
static bool
wmr_controller_base_predict_led_gate(struct xrt_device *xdev,
                                     timepoint_ns frame_mono_ns,
                                     const struct xrt_pose *P_xrworld_cam,
                                     const struct t_constellation_cam_calib *cam_calib,
                                     const struct xrt_vec3 *led_obj,
                                     float out_zhat[2],
                                     float out_S[4])
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	if (wcb->kalman_fusion == NULL || P_xrworld_cam == NULL || cam_calib == NULL || led_obj == NULL) {
		return false;
	}
	struct kalman_led_camera_view view = {
	    .fx = cam_calib->fx,
	    .fy = cam_calib->fy,
	    .cx = cam_calib->cx,
	    .cy = cam_calib->cy,
	    .cam_world_orient = P_xrworld_cam->orientation,
	    .cam_world_pos = P_xrworld_cam->position,
	};
	struct kalman_led_observation obs = {.led_obj = *led_obj}; // observed_px unused by the gate predictor
	const timepoint_ns fusion_ts = frame_mono_ns + wcb->ctrl_optical_td_ns;
	os_mutex_lock(&wcb->data_lock);
	bool ret = kalman_fusion_predict_led_gate(wcb->kalman_fusion, fusion_ts, &obs, &view, out_zhat, out_S);
	os_mutex_unlock(&wcb->data_lock);
	return ret;
}

static void
wmr_controller_base_push_observed_pose(struct xrt_device *xdev,
                                       timepoint_ns frame_mono_ns,
                                       const struct xrt_pose *pose)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);

	// Live HMD pose at this optical frame: the arm-reach adoption gate's origin + the body-lock reference.
	struct xrt_pose hmd_pose;
	const struct xrt_pose *hmd_world_pose = wmr_controller_query_hmd_pose(wcb, frame_mono_ns, &hmd_pose);

	os_mutex_lock(&wcb->data_lock);

	/* Snapshot the PREVIOUS optical anchor before we overwrite it, so the fusion
	 * telemetry can mirror the kalman optical-jump gate at this call site (the
	 * gate itself lives in t_tracker_kalman_fusion.cpp and is not modified). */
	struct xrt_vec3 prev_optical_pos = wcb->last_tracked_pose.position;
	timepoint_ns prev_optical_ts = wcb->last_tracked_pose_ts;

	wcb->last_tracked_pose_ts = frame_mono_ns;
	wcb->last_tracked_pose = *pose;

	/* Fusion telemetry: capture the REPORTED (ESKF) predicted pose before this optical
	 * observation is folded in, so each residual measures the actual output error:
	 *   - pos_residual = optical vs kalman-predicted position (real SLAM<->IMU drift).
	 *   - rot_residual = optical vs kalman-predicted orientation. */
	// td: align the optical capture time onto the IMU clock for fusion + the residual (default 0).
	const timepoint_ns fusion_ts = frame_mono_ns + wcb->ctrl_optical_td_ns;
	struct xrt_space_relation predicted_rel = {0};
	kalman_fusion_get_prediction(wcb->kalman_fusion, fusion_ts, &predicted_rel, hmd_world_pose);
	struct xrt_vec3 predicted_pos = predicted_rel.pose.position;
	struct xrt_quat predicted_rot = predicted_rel.pose.orientation;

	// Always hand the PnP pose to the fusion. The ESKF decides how to use it (dual-mode, see
	// t_tracker_kalman_fusion.cpp process_pose): bootstrap while untracked; reference-only while per-LED
	// is actively folding (so the same LEDs are never double-counted); and a snap-re-anchor when the
	// filter has diverged and per-LED can no longer pull it back (per-LED reprojection is too nonlinear
	// to fix a large error — only a PnP re-solve can). This is the live divergence-recovery path.
	// NULL variances => tuned defaults.
	struct xrt_pose_sample sample = {.pose = *pose, .timestamp_ns = fusion_ts};
	kalman_fusion_process_pose(wcb->kalman_fusion, &sample, NULL, NULL, 15, hmd_world_pose);

	/* Telemetry: optical observation vs the ESKF-predicted state (what get_tracked_pose reports).
	 *   - pos_residual_m  = |optical position - ESKF-predicted position|
	 *   - rot_residual_deg = angle between optical orientation and ESKF-predicted orientation */
	if (g2_telem_enabled()) {
		struct xrt_vec3 pos_diff = {pose->position.x - predicted_pos.x, pose->position.y - predicted_pos.y,
		                            pose->position.z - predicted_pos.z};
		float pos_residual_m = m_vec3_len(pos_diff);

		struct xrt_quat resid_q;
		math_quat_unrotate(&predicted_rot, &pose->orientation, &resid_q);
		math_quat_normalize(&resid_q);
		float w = resid_q.w < -1.0f ? -1.0f : (resid_q.w > 1.0f ? 1.0f : resid_q.w);
		float rot_residual_deg = (float)RAD_TO_DEG(2.0 * acosf(fabsf(w)));

		float optical_pose[7] = {pose->position.x,    pose->position.y,    pose->position.z,
		                         pose->orientation.x, pose->orientation.y, pose->orientation.z,
		                         pose->orientation.w};
		float predicted_pose[7] = {predicted_pos.x, predicted_pos.y, predicted_pos.z, predicted_rot.x,
		                           predicted_rot.y, predicted_rot.z, predicted_rot.w};

		/* Mirror the kalman_fusion outcome at this call site so the fusion outcomes
		 * have a telemetry producer. The gate constants and the residual_limit (15)
		 * match t_tracker_kalman_fusion.cpp, which owns the actual decision; this is a
		 * read-only reproduction. uint8_t outcome: 0=rejected, 1=accepted, 2=reset. */
		uint8_t outcome = 1 /* accepted */;

		/* Optical-jump gate: a candidate that moved further from the last optical
		 * anchor than a controller could physically travel is rejected (not fused).
		 * Constants mirror KalmanFusion: OPTICAL_MAX_SPEED_M_S=12, JUMP_SLACK=0.5 m. */
		const float OPTICAL_MAX_SPEED_M_S = 12.0f;
		const float OPTICAL_JUMP_SLACK_M = 0.5f;
		bool jump_rejected = false;
		if (prev_optical_ts != 0) {
			float dt = (float)((double)(frame_mono_ns - prev_optical_ts) / 1.0e9);
			if (dt < 0.0f) {
				dt = 0.0f;
			}
			float max_jump = OPTICAL_MAX_SPEED_M_S * dt + OPTICAL_JUMP_SLACK_M;
			struct xrt_vec3 jvec = {pose->position.x - prev_optical_pos.x,
			                        pose->position.y - prev_optical_pos.y,
			                        pose->position.z - prev_optical_pos.z};
			if (m_vec3_len(jvec) > max_jump) {
				jump_rejected = true;
			}
		}

		if (jump_rejected) {
			outcome = 0 /* rejected */;
				struct xrt_vec3 jvec = {pose->position.x - prev_optical_pos.x,
				                        pose->position.y - prev_optical_pos.y,
				                        pose->position.z - prev_optical_pos.z};
				g2_telem_event(wmr_controller_telem_id(wcb), (uint64_t)frame_mono_ns,
				               G2_TELEM_EV_OPTICAL_JUMP_REJECTED, m_vec3_len(jvec));
		} else if (pos_residual_m > 15.0f /* residual_limit */) {
			/* Residual too large -> the filter reset (catastrophic). */
			outcome = 2 /* reset */;
		}

		g2_telem_fusion(wmr_controller_telem_id(wcb), (uint64_t)frame_mono_ns, optical_pose, predicted_pose,
		                pos_residual_m, rot_residual_deg, outcome);
	}

	os_mutex_unlock(&wcb->data_lock);
}

static void
wmr_controller_base_push_observed_position(struct xrt_device *xdev,
                                           timepoint_ns frame_mono_ns,
                                           const struct xrt_vec3 *position,
                                           const struct xrt_vec3 *position_variance,
                                           bool refresh_optical_anchor)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	if (wcb->kalman_fusion == NULL || position == NULL) {
		return;
	}

	struct xrt_pose hmd_pose;
	const struct xrt_pose *hmd_world_pose = wmr_controller_query_hmd_pose(wcb, frame_mono_ns, &hmd_pose);
	const timepoint_ns fusion_ts = frame_mono_ns + wcb->ctrl_optical_td_ns;

	os_mutex_lock(&wcb->data_lock);
	kalman_fusion_process_position(wcb->kalman_fusion, fusion_ts, position, position_variance,
	                               hmd_world_pose, refresh_optical_anchor);
	os_mutex_unlock(&wcb->data_lock);
}

static void
wmr_controller_base_cache_pnp_pose_candidate(struct xrt_device *xdev,
                                             timepoint_ns frame_mono_ns,
                                             const struct xrt_pose *pose)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	if (wcb->kalman_fusion == NULL || pose == NULL) {
		return;
	}

	struct xrt_pose hmd_pose;
	const struct xrt_pose *hmd_world_pose = wmr_controller_query_hmd_pose(wcb, frame_mono_ns, &hmd_pose);
	const timepoint_ns fusion_ts = frame_mono_ns + wcb->ctrl_optical_td_ns;

	os_mutex_lock(&wcb->data_lock);
	kalman_fusion_cache_pnp_pose_candidate(wcb->kalman_fusion, fusion_ts, pose, hmd_world_pose);
	os_mutex_unlock(&wcb->data_lock);
}

/* The head tracker's world re-anchored (SLAM relocalization/reset, detected at the constellation
 * tracker's head-pose sample): transform the fusion's world-frame state by the rigid delta so the
 * prior lands in the new world the same camera frame the optical observations do. */
static void
wmr_controller_base_notify_world_reanchor(struct xrt_device *xdev,
                                          timepoint_ns frame_mono_ns,
                                          const struct xrt_pose *delta, const struct xrt_vec3 *new_raw_pivot,
                                          timepoint_ns publication_ns, double gyro_dps, double speed_mps)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);
	if (wcb->kalman_fusion == NULL || delta == NULL) {
		return;
	}
	(void)frame_mono_ns;
	os_mutex_lock(&wcb->data_lock);
	kalman_fusion_re_anchor_world_with_presentation(wcb->kalman_fusion, delta, new_raw_pivot,
	                                                publication_ns, gyro_dps, speed_mps);
	os_mutex_unlock(&wcb->data_lock);
}

static struct t_constellation_tracked_device_callbacks tracking_callbacks = {
    .get_estimator_prior = wmr_controller_base_get_estimator_prior,
    .validate_prior_epoch = wmr_controller_base_validate_prior_epoch,
    .predict_led_gate_from_prior = wmr_controller_base_predict_led_gate_from_prior,
    .get_led_model = wmr_controller_base_get_led_model,
    .notify_frame_received = wmr_controller_base_notify_frame,
    .push_observed_pose = wmr_controller_base_push_observed_pose,
    .push_observed_position = wmr_controller_base_push_observed_position,
	.push_observed_leds = wmr_controller_base_push_observed_leds,
	.push_brightness_update = wmr_controller_base_push_brightness_update,
	.get_pose_uncertainty = wmr_controller_base_get_pose_uncertainty,
	.get_gravity_tilt_reference = wmr_controller_base_get_gravity_tilt_reference,
	.get_predicted_pose = wmr_controller_base_get_predicted_pose,
	.get_last_optical_age_ms = wmr_controller_base_get_last_optical_age_ms,
	.predict_led_gate = wmr_controller_base_predict_led_gate,
	.cache_pnp_pose_candidate = wmr_controller_base_cache_pnp_pose_candidate,
	.notify_world_reanchor = wmr_controller_base_notify_world_reanchor,
};

void
wmr_controller_attach_to_hmd(struct wmr_controller_base *wcb, struct wmr_hmd *hmd)
{
	/* Keep the HMD's base device: its live tracked pose is the body-lock reference (out-of-view controllers
	 * ride with the head) and the arm-reach adoption gate's origin. */
	wcb->hmd_xdev = &hmd->base;
	/* Register the controller with the HMD for LED constellation tracking and LED sync timing updates */
	wcb->tracking_connection = wmr_hmd_add_tracked_controller(hmd, &wcb->base, &tracking_callbacks);
}
