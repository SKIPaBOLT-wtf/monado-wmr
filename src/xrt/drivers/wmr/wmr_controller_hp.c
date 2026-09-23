// Copyright 2020-2021, N Madsen.
// Copyright 2020-2023, Collabora, Ltd.
// Copyright 2020-2023, Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Driver for WMR Controllers.
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */
#include "math/m_api.h"

#include "util/u_device.h"
#include "util/u_trace_marker.h"
#include "util/u_var.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

#include "wmr_controller.h"

#define WMR_TRACE(ctrl, ...) U_LOG_XDEV_IFL_T(&ctrl->base.base, ctrl->base.log_level, __VA_ARGS__)
#define WMR_TRACE_HEX(ctrl, ...) U_LOG_XDEV_IFL_T_HEX(&ctrl->base.base, ctrl->base.log_level, __VA_ARGS__)
#define WMR_DEBUG(ctrl, ...) U_LOG_XDEV_IFL_D(&ctrl->base.base, ctrl->base.log_level, __VA_ARGS__)
#define WMR_DEBUG_HEX(ctrl, ...) U_LOG_XDEV_IFL_D_HEX(&ctrl->base.base, ctrl->base.log_level, __VA_ARGS__)
#define WMR_INFO(ctrl, ...) U_LOG_XDEV_IFL_I(&ctrl->base.base, ctrl->base.log_level, __VA_ARGS__)
#define WMR_WARN(ctrl, ...) U_LOG_XDEV_IFL_W(&ctrl->base.base, ctrl->base.log_level, __VA_ARGS__)
#define WMR_ERROR(ctrl, ...) U_LOG_XDEV_IFL_E(&ctrl->base.base, ctrl->base.log_level, __VA_ARGS__)

#ifdef XRT_DOXYGEN
#define WMR_PACKED
#else
#define WMR_PACKED __attribute__((packed))
#endif

/*!
 * Indices in input list of each input.
 */
enum wmr_controller_hp_input_index
{
	WMR_CONTROLLER_INDEX_MENU_CLICK,
	WMR_CONTROLLER_INDEX_HOME_CLICK,
	WMR_CONTROLLER_INDEX_SQUEEZE_CLICK,
	WMR_CONTROLLER_INDEX_SQUEEZE_VALUE,
	WMR_CONTROLLER_INDEX_TRIGGER_VALUE,
	WMR_CONTROLLER_INDEX_THUMBSTICK_CLICK,
	WMR_CONTROLLER_INDEX_THUMBSTICK,
	WMR_CONTROLLER_INDEX_GRIP_POSE,
	WMR_CONTROLLER_INDEX_AIM_POSE,
	WMR_CONTROLLER_INDEX_X_A_CLICK,
	WMR_CONTROLLER_INDEX_Y_B_CLICK,
	/* keep as last: */
	WMR_CONTROLLER_INDEX_COUNT
};

#define SET_INPUT(wcb, INDEX, NAME)                                                                                    \
	(wcb->base.inputs[WMR_CONTROLLER_INDEX_##INDEX].name = XRT_INPUT_G2_CONTROLLER_##NAME)

/*
 *
 * Bindings
 *
 */

static struct xrt_binding_input_pair touch_inputs[19] = {
    {XRT_INPUT_TOUCH_X_CLICK, XRT_INPUT_G2_CONTROLLER_X_CLICK},
    {XRT_INPUT_TOUCH_X_TOUCH, XRT_INPUT_G2_CONTROLLER_X_CLICK},
    {XRT_INPUT_TOUCH_Y_CLICK, XRT_INPUT_G2_CONTROLLER_Y_CLICK},
    {XRT_INPUT_TOUCH_Y_TOUCH, XRT_INPUT_G2_CONTROLLER_Y_CLICK},
    {XRT_INPUT_TOUCH_MENU_CLICK, XRT_INPUT_G2_CONTROLLER_MENU_CLICK},
    {XRT_INPUT_TOUCH_MENU_CLICK, XRT_INPUT_G2_CONTROLLER_HOME_CLICK},
    {XRT_INPUT_TOUCH_A_CLICK, XRT_INPUT_G2_CONTROLLER_A_CLICK},
    {XRT_INPUT_TOUCH_A_TOUCH, XRT_INPUT_G2_CONTROLLER_A_CLICK},
    {XRT_INPUT_TOUCH_B_CLICK, XRT_INPUT_G2_CONTROLLER_B_CLICK},
    {XRT_INPUT_TOUCH_B_TOUCH, XRT_INPUT_G2_CONTROLLER_B_CLICK},
    {XRT_INPUT_TOUCH_SYSTEM_CLICK, XRT_INPUT_G2_CONTROLLER_MENU_CLICK},
    {XRT_INPUT_TOUCH_SYSTEM_CLICK, XRT_INPUT_G2_CONTROLLER_HOME_CLICK},
    {XRT_INPUT_TOUCH_SQUEEZE_VALUE, XRT_INPUT_G2_CONTROLLER_SQUEEZE_VALUE},
    {XRT_INPUT_TOUCH_TRIGGER_TOUCH, XRT_INPUT_G2_CONTROLLER_TRIGGER_VALUE},
    {XRT_INPUT_TOUCH_TRIGGER_VALUE, XRT_INPUT_G2_CONTROLLER_TRIGGER_VALUE},
    {XRT_INPUT_TOUCH_THUMBSTICK_CLICK, XRT_INPUT_G2_CONTROLLER_THUMBSTICK_CLICK},
    {XRT_INPUT_TOUCH_THUMBSTICK, XRT_INPUT_G2_CONTROLLER_THUMBSTICK},
    {XRT_INPUT_TOUCH_GRIP_POSE, XRT_INPUT_G2_CONTROLLER_GRIP_POSE},
    {XRT_INPUT_TOUCH_AIM_POSE, XRT_INPUT_G2_CONTROLLER_AIM_POSE},
};

static struct xrt_binding_output_pair touch_outputs[1] = {
    {XRT_OUTPUT_NAME_TOUCH_HAPTIC, XRT_OUTPUT_NAME_G2_CONTROLLER_HAPTIC},
};

static struct xrt_binding_input_pair simple_inputs[4] = {
    {XRT_INPUT_SIMPLE_SELECT_CLICK, XRT_INPUT_G2_CONTROLLER_TRIGGER_VALUE},
    {XRT_INPUT_SIMPLE_MENU_CLICK, XRT_INPUT_G2_CONTROLLER_MENU_CLICK},
    {XRT_INPUT_SIMPLE_GRIP_POSE, XRT_INPUT_G2_CONTROLLER_GRIP_POSE},
    {XRT_INPUT_SIMPLE_AIM_POSE, XRT_INPUT_G2_CONTROLLER_AIM_POSE},
};

static struct xrt_binding_output_pair simple_outputs[1] = {
    {XRT_OUTPUT_NAME_SIMPLE_VIBRATION, XRT_OUTPUT_NAME_G2_CONTROLLER_HAPTIC},
};

static struct xrt_binding_profile binding_profiles[2] = {
    {
        .name = XRT_DEVICE_TOUCH_CONTROLLER,
        .inputs = touch_inputs,
        .input_count = ARRAY_SIZE(touch_inputs),
        .outputs = touch_outputs,
        .output_count = ARRAY_SIZE(touch_outputs),
    },
    {
        .name = XRT_DEVICE_SIMPLE_CONTROLLER,
        .inputs = simple_inputs,
        .input_count = ARRAY_SIZE(simple_inputs),
        .outputs = simple_outputs,
        .output_count = ARRAY_SIZE(simple_outputs),
    },
};

static const struct xrt_pose P_HP_G2_left_aim_grip = {
    .orientation = {.x = 0.300705, .y = 0.000000, .z = 0.000000, .w = 0.953717},
    .position = {.x = 0.000683, .y = -0.015332, .z = 0.068270}};
static const struct xrt_pose P_HP_G2_right_aim_grip = {
    .orientation = {.x = 0.300705, .y = 0.000000, .z = 0.000000, .w = 0.953717},
    .position = {.x = -0.000683, .y = -0.015332, .z = 0.068270}};
static const struct xrt_pose P_HP_G2_left_aim = {.orientation = {.x = 0.0, .y = 0.0, .z = 0.0, .w = 1.0},
                                                 .position = {.x = 0.0, .y = 0.0, .z = 0.0}};
static const struct xrt_pose P_HP_G2_right_aim = {.orientation = {.x = 0.0, .y = 0.0, .z = 0.0, .w = 1.0},
                                                  .position = {.x = 0.0, .y = 0.0, .z = 0.0}};

/* OG WMR Controller inputs struct */
struct wmr_controller_hp_input
{
	// buttons clicked
	bool menu;
	bool home;
	bool bt_pairing;
	bool squeeze_click; // Squeeze click reported on full squeeze

	// X/Y/A/B buttons
	bool x_a;
	bool y_b;

	float trigger;
	float squeeze;

	struct
	{
		bool click;
		struct xrt_vec2 values;
	} thumbstick;

	uint8_t battery;
};
#undef WMR_PACKED

/* HP WMR Controller device struct */
struct wmr_controller_hp
{
	struct wmr_controller_base base;

	//! The last decoded packet of button data
	struct wmr_controller_hp_input last_inputs;
};

/*
 *
 * WMR Motion Controller protocol helpers
 *
 */

static inline void
vec3_from_wmr_controller_accel(const int32_t sample[3], struct xrt_vec3 *out_vec)
{
	// Reverb G1 observation: 1g is approximately 490,000.
	// @todo: Confirm the scale is correct

	out_vec->x = (float)sample[0] / (98000 / 2);
	out_vec->y = (float)sample[1] / (98000 / 2);
	out_vec->z = (float)sample[2] / (98000 / 2);
}


static inline void
vec3_from_wmr_controller_gyro(const int32_t sample[3], struct xrt_vec3 *out_vec)
{
	// @todo: Confirm the scale is correct
	out_vec->x = (float)sample[0] * 0.00001f;
	out_vec->y = (float)sample[1] * 0.00001f;
	out_vec->z = (float)sample[2] * 0.00001f;
}

static bool
wmr_controller_hp_packet_parse(struct wmr_controller_hp *ctrl,
                               const unsigned char *buffer,
                               size_t len,
                               struct wmr_controller_base_imu_sample *imu_sample,
                               bool *out_imu_valid)
{
	struct wmr_controller_hp_input *last_input = &ctrl->last_inputs;
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(ctrl);

	if (len != 44) {
		U_LOG_IFL_E(wcb->log_level, "WMR Controller: unexpected message length: %zd", len);
		return false;
	}

	const unsigned char *p = buffer;

	// Read buttons
	uint8_t buttons = read8(&p);
	last_input->thumbstick.click = buttons & 0x01;
	last_input->home = buttons & 0x02;
	last_input->menu = buttons & 0x04;
	last_input->squeeze_click = buttons & 0x08; // squeeze-click
	last_input->bt_pairing = buttons & 0x20;

	// Read thumbstick coordinates (12 bit resolution)
	int16_t stick_x = read8(&p);
	uint8_t nibbles = read8(&p);
	stick_x += ((nibbles & 0x0F) << 8);
	int16_t stick_y = (nibbles >> 4);
	stick_y += (read8(&p) << 4);

	last_input->thumbstick.values.x = (float)(stick_x - 0x07FF) / 0x07FF;
	if (last_input->thumbstick.values.x > 1.0f) {
		last_input->thumbstick.values.x = 1.0f;
	} else if (last_input->thumbstick.values.x < -1.0f) {
		last_input->thumbstick.values.x = -1.0f;
	} else if (fabs(last_input->thumbstick.values.x) < wcb->thumbstick_deadzone) {
		last_input->thumbstick.values.x = 0.0f;
	}

	last_input->thumbstick.values.y = (float)(stick_y - 0x07FF) / 0x07FF;
	if (last_input->thumbstick.values.y > 1.0f) {
		last_input->thumbstick.values.y = 1.0f;
	} else if (last_input->thumbstick.values.y < -1.0f) {
		last_input->thumbstick.values.y = -1.0f;
	} else if (fabs(last_input->thumbstick.values.y) < wcb->thumbstick_deadzone) {
		last_input->thumbstick.values.y = 0.0f;
	}

	// Read trigger value (0x00 - 0xFF)
	last_input->trigger = (float)read8(&p) / 0xFF;

	/* On OG these are touchpad values, but on HP it's
	 * squeeze value and A_X/B_Y click */
	last_input->squeeze = (float)read8(&p) / 0xFF;

	buttons = read8(&p);
	last_input->x_a = buttons & 0x02;
	last_input->y_b = buttons & 0x01;

	last_input->battery = read8(&p);

	int32_t acc[3];
	acc[0] = read24(&p); // x
	acc[1] = read24(&p); // y
	acc[2] = read24(&p); // z

	imu_sample->temperature = read16(&p);

	int32_t gyro[3];
	gyro[0] = read24(&p);
	gyro[1] = read24(&p);
	gyro[2] = read24(&p);
	imu_sample->timestamp_ticks = (uint32_t)read32(&p);

	// HP status-only reports in low-power mode have all six raw IMU channels exactly zero.
	// Test before calibration: factory bias offsets would turn this sentinel into false motion.
	// A real sample with only zero gyro or only zero acceleration remains a measurement.
	*out_imu_valid = acc[0] != 0 || acc[1] != 0 || acc[2] != 0 || gyro[0] != 0 || gyro[1] != 0 || gyro[2] != 0;
	if (!*out_imu_valid) {
		return true; // buttons, trigger and hardware timestamp are still a valid status update
	}

	vec3_from_wmr_controller_accel(acc, &imu_sample->acc);
	math_matrix_3x3_transform_vec3(&wcb->config.sensors.accel.mix_matrix, &imu_sample->acc, &imu_sample->acc);
	math_vec3_accum(&wcb->config.sensors.accel.bias_offsets, &imu_sample->acc);
	math_quat_rotate_vec3(&wcb->config.sensors.transforms.P_oxr_acc.orientation, &imu_sample->acc,
	                      &imu_sample->acc);

	vec3_from_wmr_controller_gyro(gyro, &imu_sample->gyro);
	math_matrix_3x3_transform_vec3(&wcb->config.sensors.gyro.mix_matrix, &imu_sample->gyro, &imu_sample->gyro);
	math_vec3_accum(&wcb->config.sensors.gyro.bias_offsets, &imu_sample->gyro);
	math_quat_rotate_vec3(&wcb->config.sensors.transforms.P_oxr_gyr.orientation, &imu_sample->gyro,
	                      &imu_sample->gyro);

	/* Todo: More decoding here
	    read16(&p); // Unknown. Seems to depend on controller orientation (probably mag)
	    read32(&p); // Unknown.
	    read16(&p); // Unknown. Device state, etc.
	    read16(&p);
	    read16(&p);
	*/

	return true;
}

static bool
handle_input_packet(struct wmr_controller_base *wcb, uint64_t time_ns, uint8_t *buffer, uint32_t buf_size)
{
	struct wmr_controller_hp *ctrl = (struct wmr_controller_hp *)(wcb);
	struct wmr_controller_base_imu_sample imu_sample = {0};
	bool imu_valid = false;

	bool b = wmr_controller_hp_packet_parse(ctrl, buffer, buf_size, &imu_sample, &imu_valid);
	if (b) {
		wmr_controller_base_imu_sample(wcb, &imu_sample, (timepoint_ns)time_ns, imu_valid);
	}

	return b;
}

static xrt_result_t
wmr_controller_hp_update_inputs(struct xrt_device *xdev)
{
	DRV_TRACE_MARKER();

	struct wmr_controller_hp *ctrl = (struct wmr_controller_hp *)(xdev);
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);

	os_mutex_lock(&wcb->data_lock);

	struct xrt_input *xrt_inputs = xdev->inputs;
	struct wmr_controller_hp_input *cur_inputs = &ctrl->last_inputs;

	xrt_inputs[WMR_CONTROLLER_INDEX_MENU_CLICK].value.boolean = cur_inputs->menu;
	xrt_inputs[WMR_CONTROLLER_INDEX_HOME_CLICK].value.boolean = cur_inputs->home;
	xrt_inputs[WMR_CONTROLLER_INDEX_X_A_CLICK].value.boolean = cur_inputs->x_a;
	xrt_inputs[WMR_CONTROLLER_INDEX_Y_B_CLICK].value.boolean = cur_inputs->y_b;
	xrt_inputs[WMR_CONTROLLER_INDEX_SQUEEZE_CLICK].value.boolean = cur_inputs->squeeze;
	xrt_inputs[WMR_CONTROLLER_INDEX_SQUEEZE_VALUE].value.vec1.x = cur_inputs->squeeze;
	xrt_inputs[WMR_CONTROLLER_INDEX_TRIGGER_VALUE].value.vec1.x = cur_inputs->trigger;
	xrt_inputs[WMR_CONTROLLER_INDEX_THUMBSTICK_CLICK].value.boolean = cur_inputs->thumbstick.click;
	xrt_inputs[WMR_CONTROLLER_INDEX_THUMBSTICK].value.vec2 = cur_inputs->thumbstick.values;

	os_mutex_unlock(&wcb->data_lock);

	return XRT_SUCCESS;
}

static void
wmr_controller_hp_destroy(struct xrt_device *xdev)
{
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(xdev);

	wmr_controller_base_deinit(wcb);
	free(wcb);
}

struct wmr_controller_base *
wmr_controller_hp_create(struct wmr_controller_connection *conn,
                         enum xrt_device_type controller_type,
                         enum u_logging_level log_level)
{
	DRV_TRACE_MARKER();

	enum u_device_alloc_flags flags = U_DEVICE_ALLOC_TRACKING_NONE;
	struct wmr_controller_hp *ctrl =
	    U_DEVICE_ALLOCATE(struct wmr_controller_hp, flags, WMR_CONTROLLER_INDEX_COUNT, 1);
	struct wmr_controller_base *wcb = (struct wmr_controller_base *)(ctrl);

	if (!wmr_controller_base_init(wcb, conn, controller_type, log_level, wmr_controller_hp_destroy)) {
		wmr_controller_hp_destroy(&wcb->base);
		return NULL;
	}

	wcb->handle_input_packet = handle_input_packet;

	// Only set those we want to overwrite.
	wcb->base.update_inputs = wmr_controller_hp_update_inputs;
	wcb->base.name = XRT_DEVICE_HP_REVERB_G2_CONTROLLER;

	if (controller_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER) {
		snprintf(wcb->base.str, ARRAY_SIZE(wcb->base.str), "HP Reverb G2 Left Controller");
		wcb->P_aim_grip = P_HP_G2_left_aim_grip;
		wcb->P_aim = P_HP_G2_left_aim;
	} else {
		snprintf(wcb->base.str, ARRAY_SIZE(wcb->base.str), "HP Reverb G2 Right Controller");
		wcb->P_aim_grip = P_HP_G2_right_aim_grip;
		wcb->P_aim = P_HP_G2_right_aim;
	}

	SET_INPUT(wcb, MENU_CLICK, MENU_CLICK);
	SET_INPUT(wcb, HOME_CLICK, HOME_CLICK);
	SET_INPUT(wcb, SQUEEZE_CLICK, SQUEEZE_CLICK);
	SET_INPUT(wcb, SQUEEZE_VALUE, SQUEEZE_VALUE);
	SET_INPUT(wcb, TRIGGER_VALUE, TRIGGER_VALUE);
	SET_INPUT(wcb, THUMBSTICK_CLICK, THUMBSTICK_CLICK);
	SET_INPUT(wcb, THUMBSTICK, THUMBSTICK);
	SET_INPUT(wcb, GRIP_POSE, GRIP_POSE);
	SET_INPUT(wcb, AIM_POSE, AIM_POSE);
	if (controller_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER) {
		SET_INPUT(wcb, X_A_CLICK, X_CLICK);
		SET_INPUT(wcb, Y_B_CLICK, Y_CLICK);
	} else {
		SET_INPUT(wcb, X_A_CLICK, A_CLICK);
		SET_INPUT(wcb, Y_B_CLICK, B_CLICK);
	}

	for (uint32_t i = 0; i < wcb->base.input_count; i++) {
		wcb->base.inputs[0].active = true;
	}

	wcb->base.outputs[0].name = XRT_OUTPUT_NAME_WMR_HAPTIC;

	wcb->base.binding_profiles = binding_profiles;
	wcb->base.binding_profile_count = ARRAY_SIZE(binding_profiles);

	u_var_add_gui_header(wcb, NULL, "Controls");
	u_var_add_bool(wcb, &ctrl->last_inputs.menu, "input.menu");
	u_var_add_bool(wcb, &ctrl->last_inputs.home, "input.home");
	u_var_add_bool(wcb, &ctrl->last_inputs.bt_pairing, "input.bt_pairing");
	u_var_add_bool(wcb, &ctrl->last_inputs.squeeze_click, "input.squeeze.click");
	u_var_add_f32(wcb, &ctrl->last_inputs.squeeze, "input.squeeze.value");
	u_var_add_f32(wcb, &ctrl->last_inputs.trigger, "input.trigger");
	u_var_add_u8(wcb, &ctrl->last_inputs.battery, "input.battery");
	u_var_add_bool(wcb, &ctrl->last_inputs.thumbstick.click, "input.thumbstick.click");
	u_var_add_f32(wcb, &wcb->thumbstick_deadzone, "thumbstick deadzone");
	u_var_add_f32(wcb, &ctrl->last_inputs.thumbstick.values.x, "input.thumbstick.values.y");
	u_var_add_f32(wcb, &ctrl->last_inputs.thumbstick.values.y, "input.thumbstick.values.x");
	if (controller_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER) {
		u_var_add_bool(wcb, &ctrl->last_inputs.x_a, "input.x");
		u_var_add_bool(wcb, &ctrl->last_inputs.y_b, "input.y");
	} else {
		u_var_add_bool(wcb, &ctrl->last_inputs.x_a, "input.a");
		u_var_add_bool(wcb, &ctrl->last_inputs.y_b, "input.b");
	}

	return wcb;
}
