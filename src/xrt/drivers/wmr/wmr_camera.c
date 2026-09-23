// Copyright 2021, Jan Schmidt
// Copyright 2021, Philipp Zabel
// Copyright 2021, Jakob Bornecrantz
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  WMR camera interface
 * @author Jan Schmidt <jan@centricular.com>
 * @author Philipp Zabel <philipp.zabel@gmail.com>
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @ingroup drv_wmr
 */

#include "math/m_api.h"

#include "os/os_threading.h"
#include "os/os_time.h"
#include "xrt/xrt_byte_order.h"

#include "util/u_autoexpgain.h"
#include "util/u_debug.h"
#include "util/u_var.h"
#include "util/u_sink.h"
#include "util/u_frame.h"
#include "util/u_frame_ts_guard.h"
#include "util/u_g2_telemetry.h"
#include "util/u_time.h"
#include "util/u_trace_marker.h"

#include "wmr_config.h"
#include "wmr_protocol.h"
#include "wmr_camera.h"

#include <libusb.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

//! Specifies whether the user wants to enable autoexposure from the start.
DEBUG_GET_ONCE_BOOL_OPTION(wmr_autoexposure, "WMR_AUTOEXPOSURE", true)

//! Specifies whether the user wants to use the same exp/gain values for all cameras
DEBUG_GET_ONCE_BOOL_OPTION(wmr_unify_expgain, "WMR_UNIFY_EXPGAIN", false)

//! Optional, single local diagnostic image after autoexposure has settled.
DEBUG_GET_ONCE_OPTION(wmr_camera_snapshot, "WMR_CAMERA_SNAPSHOT", NULL)

static int
update_expgain(struct wmr_camera *cam, struct xrt_frame **frames);
static int
wmr_camera_set_ctrl_exposure_gain(struct wmr_camera *cam, uint8_t camera_id, uint16_t exposure, uint8_t gain);

/*
 *
 * Defines and structs.
 *
 */

#define WMR_CAM_TRACE(c, ...) U_LOG_IFL_T((c)->log_level, __VA_ARGS__)
#define WMR_CAM_DEBUG(c, ...) U_LOG_IFL_D((c)->log_level, __VA_ARGS__)
#define WMR_CAM_INFO(c, ...) U_LOG_IFL_I((c)->log_level, __VA_ARGS__)
#define WMR_CAM_WARN(c, ...) U_LOG_IFL_W((c)->log_level, __VA_ARGS__)
#define WMR_CAM_ERROR(c, ...) U_LOG_IFL_E((c)->log_level, __VA_ARGS__)

#define CAM_ENDPOINT 0x05

#define NUM_XFERS 9

#define WMR_CAMERA_CMD_GAIN 0x80
#define WMR_CAMERA_CMD_ON 0x81
#define WMR_CAMERA_CMD_OFF 0x82

#define DEFAULT_SLAM_EXPOSURE 6000
#define DEFAULT_SLAM_GAIN 127
#define DEFAULT_CTRL_EXPOSURE 0x0190
/* Analog gain register for the short controller/LED exposure slot, in 1/16 fine-gain units (valid range
 * WMR_MIN_GAIN 16 .. WMR_MAX_GAIN 255, so 16 = the unity floor). 16 was the long-standing operating point
 * every DN-denominated tracker constant was calibrated at; B3 (results/b3-signal-design-20260704) moved
 * production to 32 (m = 2): LED contrast scales x2 against the blobwatch admission margin, which stays
 * FIXED in DN by design, eliminating the measured near-extinction shoulder and recovering the
 * admission-censored top-edge band, at ~1-2 % clipping cost. The tracker's DN constants follow the
 * commanded gain via the blobwatch gain law (blobwatch_dim_noise_k), plumbed through
 * t_constellation_camera_group::ctrl_gain. */
#define DEFAULT_CTRL_GAIN 32

#define WMR_DEBUG_SINK_SLAM 0
#define WMR_DEBUG_SINK_CONTROLLER 1

struct wmr_camera_active_cmd
{
	__le32 magic;
	__le32 len;
	__le32 cmd;
} __attribute__((packed));

struct wmr_camera_gain_cmd
{
	__le32 magic;
	__le32 len;
	__le16 cmd;
	__le16 camera_id;
	__le16 exposure;   //!< observed 60 to 6000 (but supports up to ~9000)
	__le16 gain;       //!< observed 16 to 255
	__le16 camera_id2; //!< same as camera_id
} __attribute__((packed));

struct wmr_camera
{
	libusb_context *ctx;
	libusb_device_handle *dev;

	bool running;

	struct os_thread_helper usb_thread;
	int usb_complete;

	struct wmr_camera_config tcam_confs[WMR_MAX_CAMERAS]; //!< Configs for tracking cameras
	int tcam_count;                                       //!< Number of tracking cameras
	int slam_cam_count;                                   //!< Number of tracking cameras used for SLAM

	size_t xfer_size;
	size_t frame_xfer_size;
	uint32_t frame_width, frame_height;
	uint8_t last_seq;
	uint64_t last_frame_ts;
	uint32_t snapshot_slam_frames;

	/*! Device-clock plausibility guard over the footer start_ts. One guard for the whole
	 * stream: SLAM and controller transfers share one crystal-monotonic device clock
	 * (11.1 ms frame slots), so their interleaved timeline is strictly increasing. */
	struct u_frame_ts_guard device_ts_guard;
	uint64_t xfer_drop_count;              //!< Transfers dropped at the parse boundary
	timepoint_ns last_xfer_drop_warn_mono; //!< Rate limiter for the parse-boundary drop WARN

	/* Unwrapped frame sequence number */
	uint64_t frame_sequence;

	struct libusb_transfer *xfers[NUM_XFERS];

	struct wmr_camera_expgain
	{
		bool manual_control; //!< Whether to control exp/gain manually or with aeg
		uint16_t last_exposure, exposure;
		uint8_t last_gain, gain;
		struct u_var_draggable_u16 exposure_ui; //! Widget to control `exposure` value
		struct u_autoexpgain *aeg;
	} ceg[WMR_MAX_CAMERAS]; //!< Camera exposure-gain control
	bool unify_expgains;    //!< Whether to use the same exposure/gain values for all cameras

	struct u_sink_debug debug_sinks[2];

	struct xrt_frame_sink *slam_cam_sinks[WMR_MAX_CAMERAS]; //!< Downstream sinks to push SLAM tracking frames to
	struct xrt_frame_sink *controller_cam_sink; //!< Downstream sink to push controller tracking frames to

	enum u_logging_level log_level;
};


/*
 *
 * Helper functions.
 *
 */

/* Some WMR headsets use 616538 byte transfers. HP G2 needs 1233018 (4 cameras)
 * As a general formula, it seems we have:
 *   0x6000 byte packets. Each has a 32 byte header.
 *     packet contains frame data for each camera in turn.
 *     Each frame has an extra (first) line with metadata
 *   Then, there's an extra 26 bytes on the end.
 *
 *   F = camera frames X * (Y+1) + 26
 *   n_packets = F/(0x6000-32)
 *   leftover = F - n_packets*(0x6000-32)
 *   size = n_packets * 0x6000 + 32 + leftover,
 *
 *   so for 2 x 640x480 cameras:
 *			F = 2 * 640 * 481 + 26 = 615706
 *      n_packets = 615706 / 24544 = 25
 *      leftover = 615706 - 25 * 24544 = 2106
 *      size = 25 * 0x6000 + 32 + 2106 = 616538
 *
 *  For HP G2 = 4 x 640 * 480 cameras:
 *			F = 4 * 640 * 481 + 26 = 1231386
 *      n_packets = 1231386 / 24544 = 50
 *      leftover = 1231386 - 50 * 24544 = 4186
 *      size = 50 * 0x6000 + 32 + 4186 = 1233018
 *
 *  It would be good to test these calculations on other headsets with
 *  different camera setups.
 */
static bool
compute_frame_size(struct wmr_camera *cam)
{
	int i;
	int cams_found = 0;
	int width;
	int height;
	size_t F;
	size_t n_packets;
	size_t leftover;

	F = 26;

	libusb_device *dev = libusb_get_device(cam->dev);
	int ep_pkt_size = libusb_get_max_packet_size(dev, CAM_ENDPOINT);
	if (ep_pkt_size < 0) {
		WMR_CAM_ERROR(cam, "Failed to retrieve endpoint descriptor: result %d", ep_pkt_size);
		return false;
	}

	for (i = 0; i < cam->tcam_count; i++) {
		const struct wmr_camera_config *config = &cam->tcam_confs[i];

		WMR_CAM_DEBUG(cam, "Found head tracking camera index %d width %d height %d", i, config->roi.extent.w,
		              config->roi.extent.h);

		if (cams_found == 0) {
			width = config->roi.extent.w;
			height = config->roi.extent.h;
		} else if (height != config->roi.extent.h) {
			WMR_CAM_ERROR(cam, "Head tracking sensors have mismatched heights - %u != %u. Please report",
			              height, config->roi.extent.h);
			return false;
		} else {
			width += config->roi.extent.w;
		}

		cams_found++;
		F += config->roi.extent.w * (config->roi.extent.h + 1);
	}

	if (cams_found == 0) {
		return false;
	}

	if (width < 1280 || height < 480) {
		return false;
	}

	n_packets = F / (0x6000 - 32);
	leftover = F - n_packets * (0x6000 - 32);

	cam->frame_xfer_size = n_packets * 0x6000 + 32 + leftover;

	// Round up to a multiple of the max packet size to avoid overflows
	// in case of stalls reading things out
	size_t round_up = ep_pkt_size - (cam->frame_xfer_size % ep_pkt_size);
	cam->xfer_size = cam->frame_xfer_size + round_up;
	WMR_CAM_DEBUG(cam, "Rounding up xfer size by %zu from frame size %zu to %zu", round_up, cam->frame_xfer_size,
	              cam->xfer_size);

	cam->frame_width = width;
	cam->frame_height = height;

	WMR_CAM_INFO(cam, "WMR camera framebuffer %u x %u - %zu transfer size", cam->frame_width, cam->frame_height,
	             cam->xfer_size);

	return true;
}

static void *
wmr_cam_usb_thread(void *ptr)
{
	U_TRACE_SET_THREAD_NAME("WMR: USB-Camera");

	struct wmr_camera *cam = ptr;

	os_thread_helper_lock(&cam->usb_thread);
	while (os_thread_helper_is_running_locked(&cam->usb_thread) && !cam->usb_complete) {
		os_thread_helper_unlock(&cam->usb_thread);

		libusb_handle_events_completed(cam->ctx, &cam->usb_complete);

		os_thread_helper_lock(&cam->usb_thread);
	}

	//! @todo Think this is not needed? what condition are we waiting for?
	os_thread_helper_wait_locked(&cam->usb_thread);
	os_thread_helper_unlock(&cam->usb_thread);

	return NULL;
}

static int
send_buffer_to_device(struct wmr_camera *cam, uint8_t *buf, uint8_t len)
{
	struct libusb_transfer *xfer;
	uint8_t *data;

	xfer = libusb_alloc_transfer(0);
	if (xfer == NULL) {
		return LIBUSB_ERROR_NO_MEM;
	}

	data = malloc(len);
	if (data == NULL) {
		libusb_free_transfer(xfer);
		return LIBUSB_ERROR_NO_MEM;
	}

	memcpy(data, buf, len);

	libusb_fill_bulk_transfer(xfer, cam->dev, CAM_ENDPOINT | LIBUSB_ENDPOINT_OUT, data, len, NULL, NULL, 0);
	xfer->flags |= LIBUSB_TRANSFER_FREE_BUFFER | LIBUSB_TRANSFER_FREE_TRANSFER;

	return libusb_submit_transfer(xfer);
}

static int
set_active(struct wmr_camera *cam, bool active)
{
	struct wmr_camera_active_cmd cmd = {
	    .magic = __cpu_to_le32(WMR_MAGIC),
	    .len = __cpu_to_le32(sizeof(struct wmr_camera_active_cmd)),
	    .cmd = __cpu_to_le32(active ? WMR_CAMERA_CMD_ON : WMR_CAMERA_CMD_OFF),
	};

	return send_buffer_to_device(cam, (uint8_t *)&cmd, sizeof(cmd));
}

//! value field of G2_TELEM_EV_CAMERA_XFER_DROPPED.
enum wmr_camera_xfer_drop_reason
{
	WMR_CAMERA_XFER_DROP_FOOTER_MAGIC = 1,
	WMR_CAMERA_XFER_DROP_TS_REGRESSED = 2,
	WMR_CAMERA_XFER_DROP_TS_JUMP = 3,
};

/*! Record a transfer dropped at the parse boundary: a telemetry event so the next occurrence is
 * observable offline, plus one rate-limited WARN. @p device_ts_ns may be garbage — it is
 * recorded verbatim for forensics. */
static void
note_dropped_transfer(struct wmr_camera *cam,
                      uint64_t device_ts_ns,
                      enum wmr_camera_xfer_drop_reason reason,
                      const char *why)
{
	cam->xfer_drop_count++;
	if (g2_telem_enabled()) {
		g2_telem_event(0, device_ts_ns, G2_TELEM_EV_CAMERA_XFER_DROPPED, (float)reason);
	}
	timepoint_ns now = (timepoint_ns)os_monotonic_get_ns();
	if (now - cam->last_xfer_drop_warn_mono >= (timepoint_ns)U_TIME_1S_IN_NS) {
		cam->last_xfer_drop_warn_mono = now;
		WMR_CAM_WARN(cam, "Dropping camera transfer (%s), device ts %" PRIu64 " ns; %" PRIu64 " dropped so far",
		             why, device_ts_ns, cam->xfer_drop_count);
	}
}

static void LIBUSB_CALL
img_xfer_cb(struct libusb_transfer *xfer)
{
	DRV_TRACE_MARKER();

	struct wmr_camera *cam = xfer->user_data;
	bool resubmitted = false;

	if (xfer->status != LIBUSB_TRANSFER_COMPLETED) {
		WMR_CAM_DEBUG(cam, "Camera transfer completed with status: %s (%u)", libusb_error_name(xfer->status),
		              xfer->status);
		goto out;
	}

	if ((size_t)xfer->actual_length < cam->frame_xfer_size) {
		WMR_CAM_DEBUG(cam, "Camera transfer only delivered %d bytes of %zu per frame", xfer->actual_length,
		              cam->frame_xfer_size);
		goto out;
	}

	WMR_CAM_TRACE(cam, "Camera transfer complete - %d bytes of %d", xfer->actual_length, xfer->length);

	/* Convert the output into frames and send them off to debug / tracking */
	struct xrt_frame *xf = NULL;

	/* There's always one extra line of pixels with exposure info */
	u_frame_create_one_off(XRT_FORMAT_L8, cam->frame_width, cam->frame_height + 1, &xf);

	const uint8_t *src = xfer->buffer;

	uint8_t *dst = xf->data;
	size_t dst_remain = xf->size;
	const size_t chunk_size = 0x6000 - 32;

	DRV_TRACE_BEGIN(copy_to_frame);
	while (dst_remain >= 0x20) {
		const size_t to_copy = dst_remain > chunk_size ? chunk_size : dst_remain;

		/* 32 byte header seems to contain:
		 *   __be32 magic = "Dlo+"
		 *   __le32 frame_ctr;
		 *   __le32 slice_ctr;
		 *   __u8 unknown[20]; - binary block where all bytes are different each slice,
		 *                       but repeat every 8 slices. They're different each boot
		 *                       of the headset. Might just be uninitialised memory?
		 */
		uint32_t magic = __le32_to_cpu(*(__le32 *)(src));
		if (magic != WMR_MAGIC) {
			WMR_CAM_WARN(cam, "Invalid frame magic (got %x, expected %x). Dropping", magic, WMR_MAGIC);
			goto drop_frame;
		}
		src += 0x20;

		memcpy(dst, src, to_copy);
		src += to_copy;
		dst += to_copy;
		dst_remain -= to_copy;
	}
	DRV_TRACE_END(copy_to_frame);

	/* There should be exactly a 26 byte footer left over if we completely consumed the right amount of data */
	if (xfer->buffer + cam->frame_xfer_size - src != 26) {
		WMR_CAM_WARN(cam, "Invalid frame. Dropping");
		goto drop_frame;
	}

	/* Validate the footer (layout: wmr_camera_xfer_footer_parse) before ANY of its fields is
	 * used — a garbage footer's frametype would even select the pipeline. Both recorded
	 * garbage transfers (results/forensics-20260709/REPORT.md ITEM B) carried image pixels in
	 * the footer region while every chunk magic and the footer position checked out. */
	struct wmr_camera_xfer_footer footer;
	if (!wmr_camera_xfer_footer_parse(src, &footer)) {
		note_dropped_transfer(cam, footer.start_ts_ticks * WMR_MS_HOLOLENS_NS_PER_TICK,
		                      WMR_CAMERA_XFER_DROP_FOOTER_MAGIC, "invalid footer magic");
		goto drop_frame;
	}

	uint64_t frame_start_ts = footer.start_ts_ticks * WMR_MS_HOLOLENS_NS_PER_TICK;
	uint64_t frame_end_ts = footer.end_ts_ticks * WMR_MS_HOLOLENS_NS_PER_TICK;
	int64_t delta = frame_end_ts - frame_start_ts;

	/* Device-ts plausibility belt behind the magic check: the same guard the vit seam runs
	 * (strictly increasing + <= 1 s step; the first transfer seeds it; a dropped candidate
	 * can never poison the reference or stall the stream). */
	enum u_frame_ts_guard_verdict ts_verdict = u_frame_ts_guard_check(&cam->device_ts_guard, (int64_t)frame_start_ts);
	if (ts_verdict != U_FRAME_TS_GUARD_ACCEPT) {
		bool regressed = ts_verdict == U_FRAME_TS_GUARD_DROP_REGRESSED;
		note_dropped_transfer(cam, frame_start_ts,
		                      regressed ? WMR_CAMERA_XFER_DROP_TS_REGRESSED : WMR_CAMERA_XFER_DROP_TS_JUMP,
		                      regressed ? "device timestamp regressed" : "implausible device timestamp jump");
		goto drop_frame;
	}

	/* frametype 0 is SLAM, frametype 2 is controller tracking */
	bool slam_tracking_frame = (footer.frametype == WMR_FRAMETYPE_SLAM);

	WMR_CAM_TRACE(cam,
	              "Frame start TS %" PRIu64 " (%" PRIi64 " since last) end %" PRIu64 " dt %" PRIi64
	              " unknown %u %u frame type %u",
	              frame_start_ts, frame_start_ts - cam->last_frame_ts, frame_end_ts, delta, footer.ctr1,
	              footer.unknown0, footer.frametype);

	/* Read values from the pixel header */
	uint16_t exposure = xf->data[6] << 8 | xf->data[7];
	uint8_t seq = xf->data[89];
	uint8_t seq_delta = seq - cam->last_seq;
	uint64_t prev_frame_sequence = cam->frame_sequence;

	/* Extend the sequence number to 64-bits */
	cam->frame_sequence += seq_delta;
	uint64_t source_delta = cam->frame_sequence - prev_frame_sequence;

	WMR_CAM_TRACE(cam, "Camera frame seq %u (prev %u) -> frame %" PRIu64 " - exposure %u", seq, cam->last_seq,
	              cam->frame_sequence, exposure);

	xf->source_sequence = cam->frame_sequence;
	/* Pass a mid-point timestamp for SLAM (why?). Pass
	 * the start timestamp for the controller tracking though,
	 * as those frames have really short exposures */
	if (slam_tracking_frame) {
		xf->timestamp = frame_start_ts + delta / 2;
	} else {
		xf->timestamp = frame_start_ts;
	}
	xf->source_timestamp = frame_start_ts;

	cam->last_frame_ts = frame_start_ts;
	cam->last_seq = seq;

	if (!slam_tracking_frame && prev_frame_sequence != 0 && source_delta > 2 && g2_telem_enabled()) {
		g2_telem_event(0, xf->timestamp, G2_TELEM_EV_CAMERA_SOURCE_DELTA, (float)source_delta);
	}

	/*
	 * The USB buffer has been copied into @p xf. Resubmit before debug/tracker sinks run so downstream
	 * processing cannot delay camera intake.
	 */
	if (libusb_submit_transfer(xfer) == 0) {
		resubmitted = true;
	}

	const char *snapshot_path = debug_get_option_wmr_camera_snapshot();
	if (slam_tracking_frame && snapshot_path != NULL && ++cam->snapshot_slam_frames == 90) {
		FILE *image = fopen(snapshot_path, "wb");
		if (image != NULL) {
			fprintf(image, "P5\n%u %u\n255\n", xf->width, xf->height - 1);
			for (uint32_t row = 1; row < xf->height; row++) {
				fwrite(xf->data + row * xf->stride, 1, xf->width, image);
			}
			fclose(image);
			WMR_CAM_INFO(cam, "Saved diagnostic camera snapshot to %s", snapshot_path);
		} else {
			WMR_CAM_WARN(cam, "Cannot write diagnostic camera snapshot to %s", snapshot_path);
		}
	}

	/* Push to the appropriate debug output based on frame type */
	int sink_index = slam_tracking_frame ? WMR_DEBUG_SINK_SLAM : WMR_DEBUG_SINK_CONTROLLER;
	if (u_sink_debug_is_active(&cam->debug_sinks[sink_index])) {
		u_sink_debug_push_frame(&cam->debug_sinks[sink_index], xf);
	}

	// Push to sinks
	if (slam_tracking_frame) {
		DRV_TRACE_IDENT(push_to_sinks);

		// Tracking frames usually come at ~30fps
		struct xrt_frame *frames[WMR_MAX_CAMERAS] = {NULL};
		for (int i = 0; i < cam->slam_cam_count; i++) {
			u_frame_create_roi(xf, cam->tcam_confs[i].roi, &frames[i]);
		}

		update_expgain(cam, frames);

		for (int i = 0; i < cam->slam_cam_count; i++) {
			xrt_sink_push_frame(cam->slam_cam_sinks[i], frames[i]);
		}

		for (int i = 0; i < cam->slam_cam_count; i++) {
			xrt_frame_reference(&frames[i], NULL);
		}
	} else if (cam->controller_cam_sink != NULL) {
		/* @todo: Build a frame bundle and push that instead */
		xrt_sink_push_frame(cam->controller_cam_sink, xf);
	}

drop_frame:
	xrt_frame_reference(&xf, NULL);

out:
	if (!resubmitted) {
		libusb_submit_transfer(xfer);
	}
}


/*
 *
 * 'Exported' functions.
 *
 */

void
wmr_camera_get_ctrl_exposure_gain(uint16_t *out_exposure, uint16_t *out_gain)
{
	uint16_t exposure = DEFAULT_CTRL_EXPOSURE;
	uint16_t gain = DEFAULT_CTRL_GAIN;
	const char *exp_env = getenv("G2_CTRL_EXPOSURE");
	const char *gain_env = getenv("G2_CTRL_GAIN");
	if (exp_env && *exp_env) {
		exposure = (uint16_t)atoi(exp_env);
	}
	if (gain_env && *gain_env) {
		/* The register is 8-bit; truncate exactly like the programming path so the value the
		 * tracker denominates its constants in is the value the sensor actually runs. */
		gain = (uint8_t)atoi(gain_env);
	}
	if (out_exposure != NULL) {
		*out_exposure = exposure;
	}
	if (out_gain != NULL) {
		*out_gain = gain;
	}
}

struct wmr_camera *
wmr_camera_open(struct wmr_camera_open_config *config)
{
	DRV_TRACE_MARKER();

	struct wmr_camera *cam = calloc(1, sizeof(struct wmr_camera));
	int res;
	int i;

	cam->tcam_count = config->tcam_count;
	cam->slam_cam_count = config->slam_cam_count;
	cam->log_level = config->log_level;

	for (int i = 0; i < cam->tcam_count; i++) {
		cam->tcam_confs[i] = *config->tcam_confs[i];
		cam->slam_cam_sinks[i] = config->tcam_sinks[i];
	}

	cam->controller_cam_sink = config->controller_cam_sink;

	u_frame_ts_guard_init(&cam->device_ts_guard);

	if (os_thread_helper_init(&cam->usb_thread) != 0) {
		WMR_CAM_ERROR(cam, "Failed to initialise threading");
		wmr_camera_free(cam);
		return NULL;
	}

	res = libusb_init(&cam->ctx);
	if (res < 0) {
		goto fail;
	}

	struct xrt_prober_device *dev_holo = config->dev_holo;
	cam->dev = libusb_open_device_with_vid_pid(cam->ctx, dev_holo->vendor_id, dev_holo->product_id);
	if (cam->dev == NULL) {
		goto fail;
	}

	res = libusb_claim_interface(cam->dev, 3);
	if (res < 0) {
		goto fail;
	}

	cam->usb_complete = 0;
	if (os_thread_helper_start(&cam->usb_thread, wmr_cam_usb_thread, cam) != 0) {
		WMR_CAM_ERROR(cam, "Failed to start camera USB thread");
		goto fail;
	}

	for (i = 0; i < NUM_XFERS; i++) {
		cam->xfers[i] = libusb_alloc_transfer(0);
		if (cam->xfers[i] == NULL) {
			WMR_CAM_ERROR(cam, "Failed to allocate transfer %d", i);
			res = LIBUSB_ERROR_NO_MEM;
			goto fail;
		}
	}

	bool enable_aeg = debug_get_bool_option_wmr_autoexposure();
	int frame_delay = 3; // WMR takes about three frames until the cmd changes the image
	cam->unify_expgains = debug_get_bool_option_wmr_unify_expgain();

	for (int i = 0; i < cam->tcam_count; i++) {
		struct wmr_camera_expgain *ceg = &cam->ceg[i];
		ceg->manual_control = false;
		ceg->last_exposure = DEFAULT_SLAM_EXPOSURE;
		ceg->exposure = DEFAULT_SLAM_EXPOSURE;
		ceg->last_gain = DEFAULT_SLAM_GAIN;
		ceg->gain = DEFAULT_SLAM_GAIN;
		ceg->exposure_ui.val = &ceg->exposure;
		ceg->exposure_ui.max = WMR_MAX_EXPOSURE;
		ceg->exposure_ui.min = WMR_MIN_EXPOSURE;
		ceg->exposure_ui.step = 25;
		ceg->aeg = u_autoexpgain_create(U_AEG_STRATEGY_TRACKING, enable_aeg, frame_delay);
	}

	// Set exposure & gain for controller tracking.
	// NB: the loop bound was `i = cam->tcam_count` (== end) so it never ran — the short
	// controller/LED exposure was never programmed, leaving controller frames at the long
	// SLAM exposure (room-bright), which starved constellation tracking. Start from 0.
	// Exposure/gain are env-overridable (G2_CTRL_EXPOSURE / G2_CTRL_GAIN) so the LED-vs-room
	// balance can be swept at runtime without a rebuild. Default 400 us / gain 32
	// (DEFAULT_CTRL_EXPOSURE / DEFAULT_CTRL_GAIN), resolved by the same helper that fills the
	// constellation tracker's gain so the commanded and denominated values cannot drift.
	uint16_t ctrl_exposure;
	uint16_t ctrl_gain;
	wmr_camera_get_ctrl_exposure_gain(&ctrl_exposure, &ctrl_gain);
	WMR_CAM_INFO(cam, "Controller-tracking exposure=%u gain=%u", ctrl_exposure, ctrl_gain);
	for (int i = 0; i < cam->tcam_count; i++) {
		const struct wmr_camera_config *config = &cam->tcam_confs[i];

		bool status =
		    wmr_camera_set_ctrl_exposure_gain(cam, config->location, ctrl_exposure, (uint8_t)ctrl_gain);
		if (status != 0) {
			WMR_CAM_ERROR(cam,
			              "Failed to set exposure and gain for controller tracking frames on camera %d", i);
		}
	}

	u_sink_debug_init(&cam->debug_sinks[WMR_DEBUG_SINK_SLAM]);
	u_sink_debug_init(&cam->debug_sinks[WMR_DEBUG_SINK_CONTROLLER]);
	u_var_add_root(cam, "WMR Camera", true);
	u_var_add_log_level(cam, &cam->log_level, "Log level");

	u_var_add_gui_header_begin(cam, NULL, "Camera Streams");
	u_var_add_sink_debug(cam, &cam->debug_sinks[WMR_DEBUG_SINK_SLAM], "SLAM Tracking Streams");
	u_var_add_sink_debug(cam, &cam->debug_sinks[WMR_DEBUG_SINK_CONTROLLER], "Controller Tracking Streams");
	u_var_add_gui_header_end(cam, NULL, NULL);

	u_var_add_gui_header_begin(cam, NULL, "Exposure and gain control");
	u_var_add_bool(cam, &cam->unify_expgains, "Use same values");

	for (int i = 0; i < cam->tcam_count; i++) {
		struct wmr_camera_expgain *ceg = &cam->ceg[i];
		char label[256] = {0};

		(void)snprintf(label, sizeof(label), "Control for camera %d", i);
		u_var_add_gui_header_begin(cam, NULL, label);

		(void)snprintf(label, sizeof(label), "[%d] Manual exposure and gain control", i);
		u_var_add_bool(cam, &ceg->manual_control, label);

		(void)snprintf(label, sizeof(label), "[%d] Exposure (usec)", i);
		u_var_add_draggable_u16(cam, &ceg->exposure_ui, label);

		(void)snprintf(label, sizeof(label), "[%d] Gain", i);
		u_var_add_u8(cam, &ceg->gain, label);

		(void)snprintf(label, sizeof(label), "[%d] ", i);
		u_autoexpgain_add_vars(ceg->aeg, cam, label);

		u_var_add_gui_header_end(cam, NULL, NULL);
	}

	u_var_add_gui_header_end(cam, NULL, "Auto exposure and gain control END");

	return cam;

fail:
	WMR_CAM_ERROR(cam, "Failed to open camera: %s", libusb_error_name(res));
	wmr_camera_free(cam);
	return NULL;
}

void
wmr_camera_free(struct wmr_camera *cam)
{
	DRV_TRACE_MARKER();

	// Stop the camera.
	wmr_camera_stop(cam);

	if (cam->ctx != NULL) {
		int i;

		os_thread_helper_lock(&cam->usb_thread);
		cam->usb_complete = 1;
		os_thread_helper_unlock(&cam->usb_thread);

		if (cam->dev != NULL) {
			libusb_close(cam->dev);
		}

		os_thread_helper_destroy(&cam->usb_thread);

		for (i = 0; i < NUM_XFERS; i++) {
			if (cam->xfers[i] == NULL) {
				continue;
			}

			libusb_free_transfer(cam->xfers[i]);
			cam->xfers[i] = NULL;
		}

		libusb_exit(cam->ctx);
		cam->ctx = NULL;
	}

	// Tidy the variable tracking.
	u_var_remove_root(cam);
	u_sink_debug_destroy(&cam->debug_sinks[WMR_DEBUG_SINK_SLAM]);
	u_sink_debug_destroy(&cam->debug_sinks[WMR_DEBUG_SINK_CONTROLLER]);

	free(cam);
}

bool
wmr_camera_start(struct wmr_camera *cam)
{
	DRV_TRACE_MARKER();

	int res = 0;

	if (!compute_frame_size(cam)) {
		WMR_CAM_WARN(cam, "Invalid config or no head tracking cameras found");
		goto fail;
	}

	res = set_active(cam, false);
	if (res < 0) {
		goto fail;
	}

	res = set_active(cam, true);
	if (res < 0) {
		goto fail;
	}

	res = update_expgain(cam, NULL);
	if (res < 0) {
		goto fail;
	}

	for (int i = 0; i < NUM_XFERS; i++) {
		uint8_t *recv_buf = malloc(cam->xfer_size);

		libusb_fill_bulk_transfer(cam->xfers[i], cam->dev, LIBUSB_ENDPOINT_IN | 5, recv_buf, cam->xfer_size,
		                          img_xfer_cb, cam, 0);
		cam->xfers[i]->flags |= LIBUSB_TRANSFER_FREE_BUFFER;

		res = libusb_submit_transfer(cam->xfers[i]);
		if (res < 0) {
			WMR_CAM_ERROR(cam, "Failed to submit transfer %d", i);
			goto fail;
		}
	}

	WMR_CAM_INFO(cam, "WMR camera started");

	return true;


fail:
	if (res < 0) {
		WMR_CAM_ERROR(cam, "Error starting camera input: %s", libusb_error_name(res));
	}

	wmr_camera_stop(cam);

	return false;
}

bool
wmr_camera_stop(struct wmr_camera *cam)
{
	DRV_TRACE_MARKER();

	int res;
	int i;

	if (!cam->running) {
		return true;
	}
	cam->running = false;

	for (i = 0; i < NUM_XFERS; i++) {
		if (cam->xfers[i] != NULL) {
			libusb_cancel_transfer(cam->xfers[i]);
		}
	}

	res = set_active(cam, false);
	if (res < 0) {
		goto fail;
	}

	WMR_CAM_INFO(cam, "WMR camera stopped");

	return true;


fail:
	if (res < 0) {
		WMR_CAM_ERROR(cam, "Error stopping camera input: %s", libusb_error_name(res));
	}

	return false;
}

static int
update_expgain(struct wmr_camera *cam, struct xrt_frame **frames)
{
	int res = 0;
	for (int i = 0; i < cam->tcam_count; i++) {
		const struct wmr_camera_config *config = &cam->tcam_confs[i];

		struct wmr_camera_expgain *ceg = &cam->ceg[i];

		if (!ceg->manual_control && frames != NULL && frames[i] != NULL) {
			if (!cam->unify_expgains || i == 0) {
				u_autoexpgain_update(ceg->aeg, frames[i]);
				ceg->exposure = (uint16_t)u_autoexpgain_get_exposure(ceg->aeg);
				ceg->gain = (uint8_t)u_autoexpgain_get_gain(ceg->aeg);
			} else {
				ceg->exposure = cam->ceg[0].exposure;
				ceg->gain = cam->ceg[0].gain;
			}
		}

		if (ceg->last_exposure == ceg->exposure && ceg->last_gain == ceg->gain) {
			continue;
		}
		ceg->last_exposure = ceg->exposure;
		ceg->last_gain = ceg->gain;

		bool status = wmr_camera_set_exposure_gain(cam, config->location, ceg->exposure, ceg->gain);
		if (status != 0) {
			WMR_CAM_ERROR(cam, "Failed to set exposure and gain for camera %d", i);
		}
		res |= status;
	}
	return res;
}

int
wmr_camera_set_exposure_gain(struct wmr_camera *cam, uint8_t camera_id, uint16_t exposure, uint8_t gain)
{
	DRV_TRACE_MARKER();

	WMR_CAM_TRACE(cam, "Setting camera %d exposure %u gain %u", camera_id, exposure, gain);
	struct wmr_camera_gain_cmd cmd = {
	    .magic = __cpu_to_le32(WMR_MAGIC),
	    .len = __cpu_to_le32(sizeof(struct wmr_camera_gain_cmd)),
	    .cmd = __cpu_to_le16(WMR_CAMERA_CMD_GAIN),
	    .camera_id = __cpu_to_le16(camera_id),
	    .exposure = __cpu_to_le16(exposure),
	    .gain = __cpu_to_le16(gain),
	    .camera_id2 = __cpu_to_le16(camera_id),
	};

	return send_buffer_to_device(cam, (uint8_t *)&cmd, sizeof(cmd));
}

static int
wmr_camera_set_ctrl_exposure_gain(struct wmr_camera *cam, uint8_t camera_id, uint16_t exposure, uint8_t gain)
{
	return wmr_camera_set_exposure_gain(cam, camera_id + 2, exposure, gain);
}
