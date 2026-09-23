// Copyright 2020,2024 Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Main driver code for @ref st_ovrd.
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @author Christoph Haag <christoph.haag@collabora.com>
 * @author Daniel Willmott <web@dan-w.com>
 * @author Moshi Turner <moshiturner@protonmail.com>
 * @author Korcan Hussein <korcan.hussein@collabora.com>
 * @ingroup st_ovrd
 */

#include <cstring>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <array>
#include <mutex>
#include <thread>
#include <vector>

#include "math/m_api.h"
#include "ovrd_log.hpp"
#include "openvr_driver.h"

extern "C" {
#include "ovrd_interface.h"

#include <math.h>

#include <math/m_space.h>
#include "os/os_time.h"
#include "util/u_debug.h"
#include "util/u_device.h"
#include "util/u_builders.h"
#include "util/u_hand_tracking.h"
#include "util/u_g2_telemetry.h"
#include "util/u_world_reanchor.h"

#include "xrt/xrt_config_have.h"
#include "xrt/xrt_space.h"
#include "xrt/xrt_system.h"
#include "xrt/xrt_defines.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_instance.h"
#include "xrt/xrt_visibility_mask.h"

#include "b_ovrd_generated_bindings.h"

#ifdef XRT_HAVE_XCB
#include <xcb/xcb.h>
#include <xcb/randr.h>
#endif
}
#include "math/m_vec3.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

//! When set, all controllers pretend to be Index controllers. Provides best
//! compatibility with legacy games due to steamvr's legacy binding for Index
//! controllers, but input mapping may be incomplete or not ideal.
DEBUG_GET_ONCE_BOOL_OPTION(emulate_index_controller, "STEAMVR_EMULATE_INDEX_CONTROLLER", false)

/*
 * Render-target scale (render-program L1). The 2026-07-03 session rebased the
 * GPU ledger (SteamVR's "serialized GPU" double-counts the compositor pass):
 * true app GPU is 4.23 ms at 140 %, modeled ~3.1 ms at 120 % and ~2.6 ms at
 * 110 % (pixels scale with (pct/100)^2). 120 is the quality-conservative step
 * toward the <=5 ms pipeline mandate; a live A/B decides whether 110 holds up
 * visually. SteamVR's per-app resolution override still applies on top.
 */
DEBUG_GET_ONCE_NUM_OPTION(scale_percentage, "XRT_COMPOSITOR_SCALE_PERCENTAGE", 120)

#define MODELNUM_LEN (XRT_DEVICE_NAME_LEN + 9) // "[Monado] "

#define OPENVR_BONE_COUNT 31

// Debug define(s), always off.
#undef DUMP_POSE
#undef DUMP_POSE_CONTROLLERS

/*
 * Some HMDs (notably Windows Mixed Reality headsets) only present their native
 * video mode on the DisplayPort connector a short time after the device is
 * activated over USB — the panel powers up and the display re-enumerates a few
 * seconds later. SteamVR's vrcompositor scans X RandR exactly once at startup
 * to find a direct-mode display matching the driver-reported HMD resolution; if
 * it scans before the native mode is enumerated, the RandR lease fails and the
 * compositor cannot start.
 *
 * To make the SteamVR driver robust, Init() blocks until the HMD's native mode
 * is present in X RandR. Fully generic: it waits for whatever resolution the
 * Monado HMD device reports, so it works for any headset; it is a fast no-op
 * for HMDs that enumerate immediately, and returns false without delay on
 * sessions with no X (Wayland/headless) so the caller proceeds regardless.
 */
static bool
ovrd_wait_for_hmd_randr_mode(uint32_t want_w, uint32_t want_h, int timeout_ms)
{
#ifdef XRT_HAVE_XCB
	const char *display_env = getenv("DISPLAY");
	xcb_connection_t *conn = xcb_connect(display_env, NULL);
	if (conn == NULL || xcb_connection_has_error(conn) != 0) {
		if (conn != NULL) {
			xcb_disconnect(conn);
		}
		return false;
	}

	xcb_screen_t *screen = xcb_setup_roots_iterator(xcb_get_setup(conn)).data;
	const int poll_ms = 100;
	bool found = false;

	for (int waited = 0; waited <= timeout_ms && !found; waited += poll_ms) {
		xcb_randr_get_screen_resources_cookie_t cookie =
		    xcb_randr_get_screen_resources(conn, screen->root);
		xcb_randr_get_screen_resources_reply_t *res =
		    xcb_randr_get_screen_resources_reply(conn, cookie, NULL);
		if (res != NULL) {
			xcb_randr_mode_info_t *modes = xcb_randr_get_screen_resources_modes(res);
			int mode_count = xcb_randr_get_screen_resources_modes_length(res);
			for (int i = 0; i < mode_count; i++) {
				if (modes[i].width == want_w && modes[i].height == want_h) {
					found = true;
					break;
				}
			}
			free(res);
		}
		if (!found) {
			os_nanosleep((int64_t)poll_ms * 1000 * 1000);
		}
	}

	xcb_disconnect(conn);
	return found;
#else
	(void)want_w;
	(void)want_h;
	(void)timeout_ms;
	return false;
#endif
}


/*
 * Controller
 */

struct MonadoInputComponent
{
	bool has_component;
	bool x;
	bool y;
};

struct SteamVRDriverControl
{
	const char *steamvr_control_path;
	vr::VRInputComponentHandle_t control_handle;
};

struct SteamVRDriverControlInput : SteamVRDriverControl
{
	enum xrt_input_type monado_input_type;
	enum xrt_input_name monado_input_name;

	struct MonadoInputComponent component;
};

struct SteamVRDriverControlOutput : SteamVRDriverControl
{
	enum xrt_output_type monado_output_type;
	enum xrt_output_name monado_output_name;
};

static void
copy_vec3(const struct xrt_vec3 *from, double *to)
{
	to[0] = from->x;
	to[1] = from->y;
	to[2] = from->z;
}

static void
copy_quat(const struct xrt_quat *from, vr::HmdQuaternion_t *to)
{
	to->x = from->x;
	to->y = from->y;
	to->z = from->z;
	to->w = from->w;
}

/*
 * World re-anchor glide (the B2 head resnap guard). A head correction delta owned by the driver
 * provider, also used by legacy devices without paired pose metadata, applied at the tracking->
 * presentation seam, AFTER every internal consumer (constellation vision, ESKF, telemetry taps)
 * has been served the raw pose. The HMD pull drives detection + decay (it has the freshest
 * relation and its IMU-derived velocities for the envelope). WMR controller pulls instead use
 * the exact correction paired with their own raw reanchor snapshot; their raw event arrives on
 * the camera thread and must never be combined with this independently sampled head delta. Design + offline validation: results/b2-resnap-design-20260704/.
 */
namespace {
struct WorldReanchorGuard
{
	std::mutex lock;
	struct u_world_reanchor wr;
	bool have_prev = false;
	timepoint_ns prev_ns = 0;
	struct xrt_pose prev_pose = {};
	double prev_gyro_dps = 0.0;

	WorldReanchorGuard()
	{
		u_world_reanchor_init(&wr);
	}
};
WorldReanchorGuard g_world_guard;

void
world_guard_pack_delta(const struct u_world_reanchor &wr, float out_delta7[7])
{
	out_delta7[0] = (float)wr.dp[0];
	out_delta7[1] = (float)wr.dp[1];
	out_delta7[2] = (float)wr.dp[2];
	out_delta7[3] = (float)wr.dq[0];
	out_delta7[4] = (float)wr.dq[1];
	out_delta7[5] = (float)wr.dq[2];
	out_delta7[6] = (float)wr.dq[3];
}
} // namespace

/*
 * Convert the resolved relation into a SteamVR DriverPose. @p presented is the pose to present —
 * the world re-anchor delta composed with rel->pose (== rel->pose bit-identically while no glide
 * is active). The velocities deliberately stay the relation's TRACKED velocities: the glide rate
 * is EXCLUDED, because the runtime extrapolates to photon time from them and including it would
 * re-inject a velocity discontinuity at the absorption instant — the very artifact being removed.
 * The angular-velocity device-space conversion keeps the RAW orientation: with
 * presented_q = dq*raw_q and the presented-world angular velocity dq*w, the device-space value
 * (presented_q)^-1 * (dq*w) == raw_q^-1 * w exactly, so using the raw orientation on the raw
 * velocity IS the correct presented-world conversion.
 */
static void
apply_pose(const struct xrt_space_relation *rel, const struct xrt_pose *presented, vr::DriverPose_t *m_pose)
{
	if ((rel->relation_flags & XRT_SPACE_RELATION_ORIENTATION_VALID_BIT) != 0) {
		copy_quat(&presented->orientation, &m_pose->qRotation);
		// A confirmed stationary hold remains a valid pose after optical tracking ages out.
		// Preserve its visibility while reporting the lack of fresh tracking honestly.
		if ((rel->relation_flags & XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT) == 0) {
			m_pose->result = vr::TrackingResult_Running_OutOfRange;
		}
	} else {
		m_pose->result = vr::TrackingResult_Running_OutOfRange;
		m_pose->poseIsValid = false;
	}

	// Forward the position whenever it is VALID, not only while TRACKED: the fusion's out-of-view
	// report (body-anchored coast, reach-clamped, follow-smoothed) is strictly better than holding the
	// last tracked pose — the silent hold is what felt like a hard ceiling above the cameras' FOV.
	if ((rel->relation_flags & XRT_SPACE_RELATION_POSITION_VALID_BIT) != 0) {
		copy_vec3(&presented->position, m_pose->vecPosition);
	}

	if ((rel->relation_flags & XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT) != 0) {
		// linear velocity in world space
		copy_vec3(&rel->linear_velocity, m_pose->vecVelocity);
	} else {
		// m_pose persists across pulls: without this, SteamVR keeps photon-extrapolating on the last
		// velocity (~0.5 m/s median at freeze entry on the 2026-06-11 capture) while the pose holds.
		m_pose->vecVelocity[0] = m_pose->vecVelocity[1] = m_pose->vecVelocity[2] = 0.0;
	}

	if ((rel->relation_flags & XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT) != 0) {
		// angular velocity reported by monado in world space,
		// expected by steamvr to be in "controller space"
		struct xrt_quat orientation_inv;
		math_quat_invert(&rel->pose.orientation, &orientation_inv);

		struct xrt_vec3 vel;
		math_quat_rotate_derivative(&orientation_inv, &rel->angular_velocity, &vel);

		copy_vec3(&vel, m_pose->vecAngularVelocity);
	} else {
		m_pose->vecAngularVelocity[0] = m_pose->vecAngularVelocity[1] = m_pose->vecAngularVelocity[2] = 0.0;
	}
}

#define OPENVR_BONE_COUNT 31

// these are currently only used for the root and wrist transforms, but are kept here as they are useful for debugging.
vr::VRBoneTransform_t rightOpenPose[OPENVR_BONE_COUNT] = {
    {{0.000000f, 0.000000f, 0.000000f, 1.000000f}, {1.000000f, -0.000000f, -0.000000f, 0.000000f}}, // Root
    {{0.034038f, 0.036503f, 0.164722f, 1.000000f}, {-0.055147f, -0.078608f, 0.920279f, -0.379296f}},
    //
    {{0.012083f, 0.028070f, 0.025050f, 1.000000f}, {0.567418f, -0.464112f, 0.623374f, -0.272106f}}, // Thumb
    {{-0.040406f, -0.000000f, 0.000000f, 1.000000f}, {0.994838f, 0.082939f, 0.019454f, 0.055130f}},
    {{-0.032517f, -0.000000f, -0.000000f, 1.000000f}, {0.974793f, -0.003213f, 0.021867f, -0.222015f}},
    {{-0.030464f, 0.000000f, 0.000000f, 1.000000f}, {1.000000f, -0.000000f, -0.000000f, 0.000000f}},
    //
    {{-0.000632f, 0.026866f, 0.015002f, 1.000000f}, {0.421979f, -0.644251f, 0.422133f, 0.478202f}}, // Index
    {{-0.074204f, 0.005002f, -0.000234f, 1.000000f}, {0.995332f, 0.007007f, -0.039124f, 0.087949f}},
    {{-0.043930f, 0.000000f, 0.000000f, 1.000000f}, {0.997891f, 0.045808f, 0.002142f, -0.045943f}},
    {{-0.028695f, -0.000000f, -0.000000f, 1.000000f}, {0.999649f, 0.001850f, -0.022782f, -0.013409f}},
    {{-0.022821f, -0.000000f, 0.000000f, 1.000000f}, {1.000000f, -0.000000f, 0.000000f, -0.000000f}},
    //
    {{-0.002177f, 0.007120f, 0.016319f, 1.000000f}, {0.541276f, -0.546723f, 0.460749f, 0.442520f}}, // Middle
    {{-0.070953f, -0.000779f, -0.000997f, 1.000000f}, {0.980294f, -0.167261f, -0.078959f, 0.069368f}},
    {{-0.043108f, -0.000000f, -0.000000f, 1.000000f}, {0.997947f, 0.018493f, 0.013192f, 0.059886f}},
    {{-0.033266f, -0.000000f, -0.000000f, 1.000000f}, {0.997394f, -0.003328f, -0.028225f, -0.066315f}},
    {{-0.025892f, 0.000000f, -0.000000f, 1.000000f}, {0.999195f, -0.000000f, 0.000000f, 0.040126f}},
    //
    {{-0.000513f, -0.006545f, 0.016348f, 1.000000f}, {0.550143f, -0.516692f, 0.429888f, 0.495548f}}, // Ring
    {{-0.065876f, -0.001786f, -0.000693f, 1.000000f}, {0.990420f, -0.058696f, -0.101820f, 0.072495f}},
    {{-0.040697f, -0.000000f, -0.000000f, 1.000000f}, {0.999545f, -0.002240f, 0.000004f, 0.030081f}},
    {{-0.028747f, 0.000000f, 0.000000f, 1.000000f}, {0.999102f, -0.000721f, -0.012693f, 0.040420f}},
    {{-0.022430f, 0.000000f, -0.000000f, 1.000000f}, {1.000000f, 0.000000f, 0.000000f, 0.000000f}},

    {{0.002478f, -0.018981f, 0.015214f, 1.000000f}, {0.523940f, -0.526918f, 0.326740f, 0.584025f}}, // Pinky
    {{-0.062878f, -0.002844f, -0.000332f, 1.000000f}, {0.986609f, -0.059615f, -0.135163f, 0.069132f}},
    {{-0.030220f, -0.000000f, -0.000000f, 1.000000f}, {0.994317f, 0.001896f, -0.000132f, 0.106446f}},
    {{-0.018187f, -0.000000f, -0.000000f, 1.000000f}, {0.995931f, -0.002010f, -0.052079f, -0.073526f}},
    {{-0.018018f, -0.000000f, 0.000000f, 1.000000f}, {1.000000f, 0.000000f, 0.000000f, 0.000000f}},

    {{0.006059f, 0.056285f, 0.060064f, 1.000000f}, {0.737238f, 0.202745f, -0.594267f, -0.249441f}}, // Aux
    {{0.040416f, -0.043018f, 0.019345f, 1.000000f}, {-0.290331f, 0.623527f, 0.663809f, 0.293734f}},
    {{0.039354f, -0.075674f, 0.047048f, 1.000000f}, {-0.187047f, 0.678062f, 0.659285f, 0.265683f}},
    {{0.038340f, -0.090987f, 0.082579f, 1.000000f}, {-0.183037f, 0.736793f, 0.634757f, 0.143936f}},
    {{0.031806f, -0.087214f, 0.121015f, 1.000000f}, {-0.003659f, 0.758407f, 0.639342f, 0.126678f}},
};

vr::VRBoneTransform_t leftOpenPose[OPENVR_BONE_COUNT] = {
    {{0.000000f, 0.000000f, 0.000000f, 1.000000f}, {1.000000f, -0.000000f, -0.000000f, 0.000000f}},   // Root
                                                                                                      //
    {{-0.034038f, 0.036503f, 0.164722f, 1.000000f}, {-0.055147f, -0.078608f, -0.920279f, 0.379296f}}, // Thumb
    {{-0.012083f, 0.028070f, 0.025050f, 1.000000f}, {0.464112f, 0.567418f, 0.272106f, 0.623374f}},
    {{0.040406f, 0.000000f, -0.000000f, 1.000000f}, {0.994838f, 0.082939f, 0.019454f, 0.055130f}},
    {{0.032517f, 0.000000f, 0.000000f, 1.000000f}, {0.974793f, -0.003213f, 0.021867f, -0.222015f}},
    {{0.030464f, -0.000000f, -0.000000f, 1.000000f}, {1.000000f, -0.000000f, -0.000000f, 0.000000f}},
    //
    {{0.000632f, 0.026866f, 0.015002f, 1.000000f}, {0.644251f, 0.421979f, -0.478202f, 0.422133f}}, // Index
    {{0.074204f, -0.005002f, 0.000234f, 1.000000f}, {0.995332f, 0.007007f, -0.039124f, 0.087949f}},
    {{0.043930f, -0.000000f, -0.000000f, 1.000000f}, {0.997891f, 0.045808f, 0.002142f, -0.045943f}},
    {{0.028695f, 0.000000f, 0.000000f, 1.000000f}, {0.999649f, 0.001850f, -0.022782f, -0.013409f}},
    {{0.022821f, 0.000000f, -0.000000f, 1.000000f}, {1.000000f, -0.000000f, 0.000000f, -0.000000f}},
    //
    {{0.002177f, 0.007120f, 0.016319f, 1.000000f}, {0.546723f, 0.541276f, -0.442520f, 0.460749f}}, // Middle
    {{0.070953f, 0.000779f, 0.000997f, 1.000000f}, {0.980294f, -0.167261f, -0.078959f, 0.069368f}},
    {{0.043108f, 0.000000f, 0.000000f, 1.000000f}, {0.997947f, 0.018493f, 0.013192f, 0.059886f}},
    {{0.033266f, 0.000000f, 0.000000f, 1.000000f}, {0.997394f, -0.003328f, -0.028225f, -0.066315f}},
    {{0.025892f, -0.000000f, 0.000000f, 1.000000f}, {0.999195f, -0.000000f, 0.000000f, 0.040126f}},

    {{0.000513f, -0.006545f, 0.016348f, 1.000000f}, {0.516692f, 0.550143f, -0.495548f, 0.429888f}}, // Ring
    {{0.065876f, 0.001786f, 0.000693f, 1.000000f}, {0.990420f, -0.058696f, -0.101820f, 0.072495f}},
    {{0.040697f, 0.000000f, 0.000000f, 1.000000f}, {0.999545f, -0.002240f, 0.000004f, 0.030081f}},
    {{0.028747f, -0.000000f, -0.000000f, 1.000000f}, {0.999102f, -0.000721f, -0.012693f, 0.040420f}},
    {{0.022430f, -0.000000f, 0.000000f, 1.000000f}, {1.000000f, 0.000000f, 0.000000f, 0.000000f}},

    {{-0.002478f, -0.018981f, 0.015214f, 1.000000f}, {0.526918f, 0.523940f, -0.584025f, 0.326740f}}, // Pinky
    {{0.062878f, 0.002844f, 0.000332f, 1.000000f}, {0.986609f, -0.059615f, -0.135163f, 0.069132f}},
    {{0.030220f, 0.000000f, 0.000000f, 1.000000f}, {0.994317f, 0.001896f, -0.000132f, 0.106446f}},
    {{0.018187f, 0.000000f, 0.000000f, 1.000000f}, {0.995931f, -0.002010f, -0.052079f, -0.073526f}},
    {{0.018018f, 0.000000f, -0.000000f, 1.000000f}, {1.000000f, 0.000000f, 0.000000f, 0.000000f}},

    {{-0.006059f, 0.056285f, 0.060064f, 1.000000f}, {0.737238f, 0.202745f, 0.594267f, 0.249441f}}, // Aux
    {{-0.040416f, -0.043018f, 0.019345f, 1.000000f}, {-0.290331f, 0.623527f, -0.663809f, -0.293734f}},
    {{-0.039354f, -0.075674f, 0.047048f, 1.000000f}, {-0.187047f, 0.678062f, -0.659285f, -0.265683f}},
    {{-0.038340f, -0.090987f, 0.082579f, 1.000000f}, {-0.183037f, 0.736793f, -0.634757f, -0.143936f}},
    {{-0.031806f, -0.087214f, 0.121015f, 1.000000f}, {-0.003659f, 0.758407f, -0.639342f, -0.126678f}},
};

template <class T, class U>
void
convert_quaternion(const T &p_quatA, U &p_quatB)
{
	p_quatB.x = p_quatA.x;
	p_quatB.y = p_quatA.y;
	p_quatB.z = p_quatA.z;
	p_quatB.w = p_quatA.w;
}

xrt_quat
apply_bone_hand_transform(xrt_quat p_rot, xrt_hand hand)
{
	std::swap(p_rot.x, p_rot.z);
	p_rot.z *= -1.f;
	if (hand == XRT_HAND_RIGHT)
		return p_rot;

	p_rot.x *= -1.f;
	p_rot.y *= -1.f;
	return p_rot;
}

void
metacarpal_joints_to_bone_transform(struct xrt_hand_joint_set *hand_joint_set,
                                    vr::VRBoneTransform_t *out_bone_transforms,
                                    xrt_hand hand)
{
	struct xrt_hand_joint_value *joint_values = hand_joint_set->values.hand_joint_set_default;

	// Apply orientations for four-finger metacarpals.
	for (int joint :
	     {XRT_HAND_JOINT_THUMB_METACARPAL, XRT_HAND_JOINT_INDEX_METACARPAL, XRT_HAND_JOINT_MIDDLE_METACARPAL,
	      XRT_HAND_JOINT_RING_METACARPAL, XRT_HAND_JOINT_LITTLE_METACARPAL}) {
		struct xrt_hand_joint_value *current_joint = &joint_values[joint];
		struct xrt_hand_joint_value *parent_joint = &joint_values[XRT_HAND_JOINT_WRIST];

		xrt_quat diff_openxr;
		// These should do the exact same things.
		xrt_quat parent_inv;
		math_quat_invert(&parent_joint->relation.pose.orientation, &parent_inv);
		math_quat_rotate(&parent_inv, &current_joint->relation.pose.orientation, &diff_openxr);
		xrt_quat diff_openvr = apply_bone_hand_transform(diff_openxr, hand);


		/**
		 * * if you try applying the metacarpal transforms without the magic quaternion, everything from the
		 * metacarpals onwards is rotated 90 degrees.
		 * In the neutral pose sample, all the metacarpals have a
		 * rotation relatively close to {w=0.5, x=0.5, y=-0.5, z=0.5} which is an Important Quaternion because
		 * it probably represents some 90 degree rotation. Maybe, and this was just a random guess, if I took
		 * the regular metacarpal orientations and rotated them by that quat, everything would work.
		 */
		xrt_quat magic_prerotate = XRT_QUAT_IDENTITY;
		magic_prerotate.w = 0.5;
		magic_prerotate.x = 0.5;
		magic_prerotate.y = -0.5;
		magic_prerotate.z = 0.5;

		if (hand == XRT_HAND_RIGHT) {
			magic_prerotate.y *= -1.f;
			magic_prerotate.x *= -1.f;
		}

		xrt_quat final_diff;
		math_quat_rotate(&magic_prerotate, &diff_openvr, &final_diff);
		convert_quaternion(final_diff, out_bone_transforms[joint].orientation);

		xrt_vec3 global_diff_from_this_to_parent =
		    m_vec3_sub(current_joint->relation.pose.position, parent_joint->relation.pose.position);

		xrt_vec3 translation_wrist_rel;
		math_quat_rotate_vec3(&parent_inv, &global_diff_from_this_to_parent, &translation_wrist_rel);

		// Y = X?
		out_bone_transforms[joint].position.v[0] = translation_wrist_rel.y;
		out_bone_transforms[joint].position.v[1] = translation_wrist_rel.x;
		out_bone_transforms[joint].position.v[2] = -translation_wrist_rel.z;
		out_bone_transforms[joint].position.v[3] = 1.f;

		if (hand == XRT_HAND_RIGHT)
			out_bone_transforms[joint].position.v[1] *= -1.f;
	}
}

void
flexion_joints_to_bone_transform(struct xrt_hand_joint_set *hand_joint_set,
                                 vr::VRBoneTransform_t *out_bone_transforms,
                                 xrt_hand hand)
{
	struct xrt_hand_joint_value *joint_values = hand_joint_set->values.hand_joint_set_default;

	// Apply orientations for four-finger pxm and onward
	int parent = -1;
	for (int joint = XRT_HAND_JOINT_THUMB_METACARPAL; joint < XRT_HAND_JOINT_COUNT; joint++) {
		if (u_hand_joint_is_metacarpal((xrt_hand_joint)joint)) {
			parent = joint;
			continue;
		}
		struct xrt_hand_joint_value *current_joint = &joint_values[joint];
		struct xrt_hand_joint_value *parent_joint = &joint_values[parent];


		xrt_quat diff_openxr;
		math_quat_unrotate(&parent_joint->relation.pose.orientation, &current_joint->relation.pose.orientation,
		                   &diff_openxr);

		xrt_quat diff_openvr = apply_bone_hand_transform(diff_openxr, hand);
		convert_quaternion(diff_openvr, out_bone_transforms[joint].orientation);
		xrt_vec3 global_diff_from_this_to_parent =
		    m_vec3_sub(current_joint->relation.pose.position, parent_joint->relation.pose.position);


		float bone_length = m_vec3_len(global_diff_from_this_to_parent);
		// OpenVR left hand has +X forward. Weird, huh?
		out_bone_transforms[joint].position = {bone_length, 0, 0, 1};

		if (hand == XRT_HAND_RIGHT)
			out_bone_transforms[joint].position.v[0] *= -1.f;

		parent = joint;
	}
}

void
hand_joint_set_to_bone_transform(struct xrt_hand_joint_set hand_joint_set,
                                 vr::VRBoneTransform_t *out_bone_transforms,
                                 xrt_hand hand)
{
	// fill bone transforms with a default open pose to manipulate later
	for (int i : {XRT_HAND_JOINT_WRIST, XRT_HAND_JOINT_PALM}) {
		out_bone_transforms[i] = hand == XRT_HAND_LEFT ? leftOpenPose[i] : rightOpenPose[i];
	}

	metacarpal_joints_to_bone_transform(&hand_joint_set, out_bone_transforms, hand);
	flexion_joints_to_bone_transform(&hand_joint_set, out_bone_transforms, hand);
}

class CDeviceDriver_Monado_Controller : public vr::ITrackedDeviceServerDriver
{
public:
	CDeviceDriver_Monado_Controller(struct xrt_instance *xinst, struct xrt_device *xdev, enum xrt_hand hand)
	    : m_xdev(xdev), m_hand(hand)
	{
		ovrd_log("Creating Controller %s\n", xdev->str);

		m_handed_controller = true;

		m_emulate_index_controller = debug_get_bool_option_emulate_index_controller();

		if (m_emulate_index_controller) {
			ovrd_log("Emulating Index Controller\n");
		} else {
			ovrd_log("Using Monado Controller profile\n");
		}

		m_unObjectId = vr::k_unTrackedDeviceIndexInvalid;
		m_pose = {};

		snprintf(m_sModelNumber, MODELNUM_LEN, "[Monado] %s", xdev->str);
		strncpy(m_sSerialNumber, xdev->serial, XRT_DEVICE_NAME_LEN);

		switch (this->m_xdev->name) {
		case XRT_DEVICE_INDEX_CONTROLLER:
			if (hand == XRT_HAND_LEFT) {
				m_render_model =
				    "{indexcontroller}valve_controller_knu_1_0_"
				    "left";
			}
			if (hand == XRT_HAND_RIGHT) {
				m_render_model =
				    "{indexcontroller}valve_controller_knu_1_0_"
				    "right";
			}
			break;
		case XRT_DEVICE_TOUCH_CONTROLLER:
			if (hand == XRT_HAND_LEFT) {
				m_render_model = "oculus_cv1_controller_left";
			}
			if (hand == XRT_HAND_RIGHT) {
				m_render_model = "oculus_cv1_controller_right";
			}
			break;
		case XRT_DEVICE_HP_REVERB_G2_CONTROLLER:
			m_render_model = hand == XRT_HAND_LEFT ? "{monado}hp_reverb_g2_left" : "{monado}hp_reverb_g2_right";
			break;
		case XRT_DEVICE_VIVE_WAND: m_render_model = "vr_controller_vive_1_5"; break;
		case XRT_DEVICE_VIVE_TRACKER_GEN1:
		case XRT_DEVICE_VIVE_TRACKER_GEN2:
		case XRT_DEVICE_VIVE_TRACKER_GEN3:
		case XRT_DEVICE_VIVE_TRACKER_TUNDRA: m_render_model = "{htc}vr_tracker_vive_1_0"; break;
		case XRT_DEVICE_FLIPVR:
			if (hand == XRT_HAND_LEFT) {
				m_render_model = "{shiftall}flipvr_controller_vc1b_left";
			}
			if (hand == XRT_HAND_RIGHT) {
				m_render_model = "{shiftall}flipvr_controller_vc1b_right";
			}
			break;
		case XRT_DEVICE_PSMV:
		case XRT_DEVICE_HYDRA:
		case XRT_DEVICE_DAYDREAM:
		case XRT_DEVICE_GENERIC_HMD:
		default: m_render_model = "locator_one_sided"; break;
		}

		ovrd_log("Render model based on Monado: %s\n", m_render_model);

		vr::VRServerDriverHost()->TrackedDeviceAdded(GetSerialNumber().c_str(),
		                                             vr::TrackedDeviceClass_Controller, this);
	}
	virtual ~CDeviceDriver_Monado_Controller() {}

	void
	AddControl(const char *steamvr_control_path,
	           enum xrt_input_name monado_input_name,
	           struct MonadoInputComponent *component)
	{
		enum xrt_input_type monado_input_type = XRT_GET_INPUT_TYPE(monado_input_name);

		SteamVRDriverControlInput in;

		in.monado_input_type = monado_input_type;
		in.steamvr_control_path = steamvr_control_path;
		in.monado_input_name = monado_input_name;
		if (component != NULL) {
			in.component = *component;
		} else {
			in.component.has_component = false;
		}

		if (monado_input_type == XRT_INPUT_TYPE_BOOLEAN) {
			vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, steamvr_control_path,
			                                            &in.control_handle);

		} else if (monado_input_type == XRT_INPUT_TYPE_VEC1_MINUS_ONE_TO_ONE) {
			vr::VRDriverInput()->CreateScalarComponent(m_ulPropertyContainer, steamvr_control_path,
			                                           &in.control_handle, vr::VRScalarType_Absolute,
			                                           vr::VRScalarUnits_NormalizedTwoSided);

		} else if (monado_input_type == XRT_INPUT_TYPE_VEC1_ZERO_TO_ONE) {
			vr::VRDriverInput()->CreateScalarComponent(m_ulPropertyContainer, steamvr_control_path,
			                                           &in.control_handle, vr::VRScalarType_Absolute,
			                                           vr::VRScalarUnits_NormalizedOneSided);

		} else if (monado_input_type == XRT_INPUT_TYPE_VEC2_MINUS_ONE_TO_ONE) {
			// 2D values are added as 2 1D values
			// usually those are [-1,1]
			vr::VRDriverInput()->CreateScalarComponent(m_ulPropertyContainer, steamvr_control_path,
			                                           &in.control_handle, vr::VRScalarType_Absolute,
			                                           vr::VRScalarUnits_NormalizedTwoSided);
		}

		m_input_controls.push_back(in);
		ovrd_log("Added input %s\n", steamvr_control_path);
	}

	void
	AddOutputControl(enum xrt_output_name monado_output_name, const char *steamvr_control_path)
	{
		SteamVRDriverControlOutput out;

		/// @todo when there are multiple output types: XRT_GET_OUTPUT_TYPE(monado_output_name);
		enum xrt_output_type monado_output_type = XRT_OUTPUT_TYPE_VIBRATION;

		out.monado_output_type = monado_output_type;
		out.steamvr_control_path = steamvr_control_path;
		out.monado_output_name = monado_output_name;

		vr::VRDriverInput()->CreateHapticComponent(m_ulPropertyContainer, out.steamvr_control_path,
		                                           &out.control_handle);

		m_output_controls.push_back(out);
		ovrd_log("Added output %s\n", steamvr_control_path);
	}

	void
	AddSkeletonControl(const char *steamvr_skeleton_name,
	                   const char *steamvr_control_path,
	                   enum xrt_input_name monado_input_name)
	{
		enum xrt_input_type monado_input_type = XRT_GET_INPUT_TYPE(monado_input_name);

		SteamVRDriverControlInput in;

		in.monado_input_type = monado_input_type;
		in.steamvr_control_path = steamvr_control_path;
		in.monado_input_name = monado_input_name;
		in.component.has_component = false;

		vr::EVRInputError err = vr::VRDriverInput()->CreateSkeletonComponent(
		    m_ulPropertyContainer, steamvr_skeleton_name, steamvr_control_path, "/pose/raw",
		    vr::VRSkeletalTracking_Full, NULL, OPENVR_BONE_COUNT, &in.control_handle);
		if (err) {
			ovrd_log("Error adding skeletal input: %i", err);
			return;
		}


		m_skeletal_input_control = in;
		ovrd_log("Added skeleton input %s\n", steamvr_control_path);
	}

	void
	AddEmulatedIndexControls()
	{
		switch (this->m_xdev->name) {
		case XRT_DEVICE_INDEX_CONTROLLER: {
			AddControl("/input/trigger/value", XRT_INPUT_INDEX_TRIGGER_VALUE, NULL);

			AddControl("/input/trigger/click", XRT_INPUT_INDEX_TRIGGER_CLICK, NULL);
			AddControl("/input/trigger/touch", XRT_INPUT_INDEX_TRIGGER_TOUCH, NULL);


			AddControl("/input/system/click", XRT_INPUT_INDEX_SYSTEM_CLICK, NULL);
			AddControl("/input/system/touch", XRT_INPUT_INDEX_SYSTEM_TOUCH, NULL);

			AddControl("/input/a/click", XRT_INPUT_INDEX_A_CLICK, NULL);
			AddControl("/input/a/touch", XRT_INPUT_INDEX_A_TOUCH, NULL);

			AddControl("/input/b/click", XRT_INPUT_INDEX_B_CLICK, NULL);
			AddControl("/input/b/touch", XRT_INPUT_INDEX_B_TOUCH, NULL);


			AddControl("/input/grip/force", XRT_INPUT_INDEX_SQUEEZE_FORCE, NULL);

			AddControl("/input/grip/value", XRT_INPUT_INDEX_SQUEEZE_VALUE, NULL);

			struct MonadoInputComponent x = {true, true, false};
			struct MonadoInputComponent y = {true, false, true};

			AddControl("/input/thumbstick/click", XRT_INPUT_INDEX_THUMBSTICK_CLICK, NULL);

			AddControl("/input/thumbstick/touch", XRT_INPUT_INDEX_THUMBSTICK_TOUCH, NULL);

			AddControl("/input/thumbstick/x", XRT_INPUT_INDEX_THUMBSTICK, &x);

			AddControl("/input/thumbstick/y", XRT_INPUT_INDEX_THUMBSTICK, &y);


			AddControl("/input/trackpad/force", XRT_INPUT_INDEX_TRACKPAD_FORCE, NULL);

			AddControl("/input/trackpad/touch", XRT_INPUT_INDEX_TRACKPAD_TOUCH, NULL);

			AddControl("/input/trackpad/x", XRT_INPUT_INDEX_TRACKPAD, &x);

			AddControl("/input/trackpad/y", XRT_INPUT_INDEX_TRACKPAD, &y);

			if (m_xdev->supported.hand_tracking) {
				ovrd_log("Enabling skeletal input as this device supports it");

				// skeletal input compatibility with games is a bit funky with any controllers
				// other than the index controller, so only do skeletal input with index
				// emulation
				const std::string str_hand = m_hand == XRT_HAND_LEFT ? "left" : "right";


				AddSkeletonControl(("/input/skeleton/" + str_hand).c_str(),
				                   ("/skeleton/hand/" + str_hand).c_str(),
				                   XRT_INPUT_HT_UNOBSTRUCTED_RIGHT);
				RunFrame();
			} else
				ovrd_log("Not enabling skeletal input as this device does not support it");


			AddOutputControl(XRT_OUTPUT_NAME_INDEX_HAPTIC, "/output/haptic");
		}

		break;
		case XRT_DEVICE_VIVE_WAND: {
			AddControl("/input/trigger/value", XRT_INPUT_VIVE_TRIGGER_VALUE, NULL);

			AddControl("/input/trigger/click", XRT_INPUT_VIVE_TRIGGER_CLICK, NULL);


			AddControl("/input/system/click", XRT_INPUT_VIVE_SYSTEM_CLICK, NULL);

			AddControl("/input/a/click", XRT_INPUT_VIVE_TRACKPAD_CLICK, NULL);

			AddControl("/input/b/click", XRT_INPUT_VIVE_MENU_CLICK, NULL);

			struct MonadoInputComponent x = {true, true, false};
			struct MonadoInputComponent y = {true, false, true};

			AddControl("/input/trackpad/touch", XRT_INPUT_VIVE_TRACKPAD_TOUCH, NULL);

			AddControl("/input/trackpad/x", XRT_INPUT_VIVE_TRACKPAD, &x);

			AddControl("/input/trackpad/y", XRT_INPUT_VIVE_TRACKPAD, &y);

			AddOutputControl(XRT_OUTPUT_NAME_VIVE_HAPTIC, "/output/haptic");
		} break;
		case XRT_DEVICE_PSMV: {

			AddControl("/input/trigger/value", XRT_INPUT_PSMV_TRIGGER_VALUE, NULL);

			AddControl("/input/trigger/click", XRT_INPUT_PSMV_MOVE_CLICK, NULL);

			AddControl("/input/system/click", XRT_INPUT_PSMV_PS_CLICK, NULL);

			AddControl("/input/a/click", XRT_INPUT_PSMV_CROSS_CLICK, NULL);

			AddControl("/input/b/click", XRT_INPUT_PSMV_SQUARE_CLICK, NULL);

			AddOutputControl(XRT_OUTPUT_NAME_PSMV_RUMBLE_VIBRATION, "/output/haptic");
		} break;
		case XRT_DEVICE_FLIPVR: {

			AddControl("/input/trigger/value", XRT_INPUT_FLIPVR_TRIGGER_VALUE, NULL);

			AddControl("/input/trigger/click", XRT_INPUT_FLIPVR_TRIGGER_CLICK, NULL);
			AddControl("/input/trigger/touch", XRT_INPUT_FLIPVR_TRIGGER_TOUCH, NULL);

			AddControl("/input/system/click", XRT_INPUT_FLIPVR_SYSTEM_CLICK, NULL);

			AddControl("/input/a/click", XRT_INPUT_FLIPVR_A_CLICK, NULL);
			AddControl("/input/a/touch", XRT_INPUT_FLIPVR_A_TOUCH, NULL);

			AddControl("/input/b/click", XRT_INPUT_FLIPVR_B_CLICK, NULL);
			AddControl("/input/b/touch", XRT_INPUT_FLIPVR_B_TOUCH, NULL);

			AddControl("/input/a/click", XRT_INPUT_FLIPVR_X_CLICK, NULL);
			AddControl("/input/a/touch", XRT_INPUT_FLIPVR_X_TOUCH, NULL);

			AddControl("/input/b/click", XRT_INPUT_FLIPVR_Y_CLICK, NULL);
			AddControl("/input/b/touch", XRT_INPUT_FLIPVR_Y_TOUCH, NULL);

			AddControl("/input/grip/value", XRT_INPUT_FLIPVR_SQUEEZE_VALUE, NULL);

			AddControl("/input/grip/click", XRT_INPUT_FLIPVR_SQUEEZE_CLICK, NULL);

			struct MonadoInputComponent x = {true, true, false};
			struct MonadoInputComponent y = {true, false, true};

			AddControl("/input/thumbstick/click", XRT_INPUT_FLIPVR_THUMBSTICK_CLICK, NULL);

			AddControl("/input/thumbstick/touch", XRT_INPUT_FLIPVR_THUMBSTICK_TOUCH, NULL);

			AddControl("/input/thumbstick/x", XRT_INPUT_FLIPVR_THUMBSTICK, &x);

			AddControl("/input/thumbstick/y", XRT_INPUT_FLIPVR_THUMBSTICK, &y);

			AddOutputControl(XRT_OUTPUT_NAME_FLIPVR_HAPTIC, "/output/haptic");
		}

		case XRT_DEVICE_TOUCH_CONTROLLER: break; // TODO
		case XRT_DEVICE_WMR_CONTROLLER: {
			AddControl("/input/trigger/value", XRT_INPUT_WMR_TRIGGER_VALUE, NULL);
			AddControl("/input/grip/click", XRT_INPUT_WMR_SQUEEZE_CLICK, NULL);

			// @todo: Does the WMR driver (and variants) need to report a synthesised click input?
			// AddControl("/input/trigger/click", XRT_INPUT_WMR_TRIGGER_CLICK, NULL);

			AddControl("/input/system/click", XRT_INPUT_WMR_HOME_CLICK, NULL);

			struct MonadoInputComponent x = {true, true, false};
			struct MonadoInputComponent y = {true, false, true};

			AddControl("/input/trackpad/click", XRT_INPUT_WMR_TRACKPAD_CLICK, NULL);
			AddControl("/input/trackpad/touch", XRT_INPUT_WMR_TRACKPAD_TOUCH, NULL);
			AddControl("/input/trackpad/x", XRT_INPUT_WMR_TRACKPAD, &x);
			AddControl("/input/trackpad/y", XRT_INPUT_WMR_TRACKPAD, &y);

			AddControl("/input/thumbstick/click", XRT_INPUT_WMR_THUMBSTICK_CLICK, NULL);
			AddControl("/input/thumbstick/x", XRT_INPUT_WMR_THUMBSTICK, &x);
			AddControl("/input/thumbstick/y", XRT_INPUT_WMR_THUMBSTICK, &y);
			break;
		}
		case XRT_DEVICE_SAMSUNG_ODYSSEY_CONTROLLER: {
			// Same as the OG controller, but with distinct input names
			AddControl("/input/trigger/value", XRT_INPUT_ODYSSEY_CONTROLLER_TRIGGER_VALUE, NULL);
			AddControl("/input/grip/click", XRT_INPUT_ODYSSEY_CONTROLLER_SQUEEZE_CLICK, NULL);

			AddControl("/input/system/click", XRT_INPUT_ODYSSEY_CONTROLLER_HOME_CLICK, NULL);

			struct MonadoInputComponent x = {true, true, false};
			struct MonadoInputComponent y = {true, false, true};

			AddControl("/input/trackpad/click", XRT_INPUT_ODYSSEY_CONTROLLER_TRACKPAD_CLICK, NULL);
			AddControl("/input/trackpad/touch", XRT_INPUT_ODYSSEY_CONTROLLER_TRACKPAD_TOUCH, NULL);
			AddControl("/input/trackpad/x", XRT_INPUT_ODYSSEY_CONTROLLER_TRACKPAD, &x);
			AddControl("/input/trackpad/y", XRT_INPUT_ODYSSEY_CONTROLLER_TRACKPAD, &y);

			AddControl("/input/thumbstick/click", XRT_INPUT_ODYSSEY_CONTROLLER_THUMBSTICK_CLICK, NULL);
			AddControl("/input/thumbstick/x", XRT_INPUT_ODYSSEY_CONTROLLER_THUMBSTICK, &x);
			AddControl("/input/thumbstick/y", XRT_INPUT_ODYSSEY_CONTROLLER_THUMBSTICK, &y);
			break;
		}
		case XRT_DEVICE_HP_REVERB_G2_CONTROLLER: {
			// The SteamVR component paths below MUST match the input
			// sources declared in the generated HP Reverb G2 input
			// profile (hp_mixed_reality_controller_profile.json):
			// /input/{trigger,grip,joystick,menu,system,x,y,a,b} and
			// /output/haptic. A path not declared in that profile is
			// silently ignored by SteamVR — which is why the joystick
			// (was wrongly /input/thumbstick), the menu button, the
			// X/Y buttons and haptics previously did nothing, leaving
			// the controller usable only as a bare tracked pose.
			AddControl("/input/trigger/value", XRT_INPUT_G2_CONTROLLER_TRIGGER_VALUE, NULL);
			AddControl("/input/grip/value", XRT_INPUT_G2_CONTROLLER_SQUEEZE_VALUE, NULL);

			AddControl("/input/menu/click", XRT_INPUT_G2_CONTROLLER_MENU_CLICK, NULL);
			AddControl("/input/system/click", XRT_INPUT_G2_CONTROLLER_HOME_CLICK, NULL);

			// G2 controllers are handed: the left has X/Y face buttons,
			// the right has A/B. Map each to its own profile path.
			if (m_hand == XRT_HAND_LEFT) {
				AddControl("/input/x/click", XRT_INPUT_G2_CONTROLLER_X_CLICK, NULL);
				AddControl("/input/y/click", XRT_INPUT_G2_CONTROLLER_Y_CLICK, NULL);
			} else {
				AddControl("/input/a/click", XRT_INPUT_G2_CONTROLLER_A_CLICK, NULL);
				AddControl("/input/b/click", XRT_INPUT_G2_CONTROLLER_B_CLICK, NULL);
			}

			struct MonadoInputComponent x = {true, true, false};
			struct MonadoInputComponent y = {true, false, true};

			AddControl("/input/joystick/click", XRT_INPUT_G2_CONTROLLER_THUMBSTICK_CLICK, NULL);
			AddControl("/input/joystick/x", XRT_INPUT_G2_CONTROLLER_THUMBSTICK, &x);
			AddControl("/input/joystick/y", XRT_INPUT_G2_CONTROLLER_THUMBSTICK, &y);

			AddOutputControl(XRT_OUTPUT_NAME_G2_CONTROLLER_HAPTIC, "/output/haptic");
			break;
		}
		case XRT_DEVICE_XBOX_CONTROLLER: break;     // TODO
		case XRT_DEVICE_VIVE_TRACKER_GEN1: break;   // TODO
		case XRT_DEVICE_VIVE_TRACKER_GEN2: break;   // TODO
		case XRT_DEVICE_VIVE_TRACKER_GEN3: break;   // TODO
		case XRT_DEVICE_VIVE_TRACKER_TUNDRA: break; // TODO
		case XRT_DEVICE_REALSENSE: break;
		case XRT_DEVICE_DEPTHAI: break;

		case XRT_DEVICE_HAND_INTERACTION: break;  // there is no hardware
		case XRT_DEVICE_GO_CONTROLLER: break;     // hardware has no haptics
		case XRT_DEVICE_DAYDREAM: break;          // hardware has no haptics
		case XRT_DEVICE_HYDRA: break;             // hardware has no haptics
		case XRT_DEVICE_SIMPLE_CONTROLLER: break; // shouldn't happen
		case XRT_DEVICE_HAND_TRACKER: break;      // shouldn't happen
		case XRT_DEVICE_GENERIC_HMD:
		case XRT_DEVICE_VIVE_PRO: break; // no
		case XRT_DEVICE_EXT_HAND_INTERACTION: break;

		default: break;
		}
	}

	static struct profile_template *
	get_profile_template(enum xrt_device_name device_name)
	{
		for (int i = 0; i < OXR_BINDINGS_PROFILE_TEMPLATE_COUNT; i++) {
			if (profile_templates[i].name == device_name)
				return &profile_templates[i];
		}
		return NULL;
	}

	void
	AddMonadoInput(struct binding_template *b)
	{
		enum xrt_input_name monado_input_name = b->input;
		const char *steamvr_path = b->steamvr_path;

		enum xrt_input_type monado_input_type = XRT_GET_INPUT_TYPE(monado_input_name);

		switch (monado_input_type) {
		case XRT_INPUT_TYPE_BOOLEAN:
		case XRT_INPUT_TYPE_VEC1_MINUS_ONE_TO_ONE:
		case XRT_INPUT_TYPE_VEC1_ZERO_TO_ONE: AddControl(steamvr_path, monado_input_name, NULL); break;

		case XRT_INPUT_TYPE_VEC2_MINUS_ONE_TO_ONE: {
			std::string xpath = std::string(steamvr_path) + std::string("/x");
			std::string ypath = std::string(steamvr_path) + std::string("/y");

			struct MonadoInputComponent x = {true, true, false};
			struct MonadoInputComponent y = {true, false, true};

			AddControl(xpath.c_str(), monado_input_name, &x);
			AddControl(ypath.c_str(), monado_input_name, &y);
		} break;
		case XRT_INPUT_TYPE_POSE:
			//! @todo how to handle poses?
		case XRT_INPUT_TYPE_HAND_TRACKING:
		case XRT_INPUT_TYPE_FACE_TRACKING:
		case XRT_INPUT_TYPE_BODY_TRACKING:
		case XRT_INPUT_TYPE_VEC3_MINUS_ONE_TO_ONE: break;
		}
	}

	void
	AddMonadoControls()
	{
		struct profile_template *p = get_profile_template(m_xdev->name);
		if (!p) {
			ovrd_log("No profile template for %s\n", m_xdev->str);
			return;
		}

		// Binding templates are generated per subaction path: every input of the
		// interaction profile appears once per hand it exists on. This device is one
		// hand, so instantiate only its own bindings — otherwise shared inputs get
		// duplicate components and side-gated ones (e.g. the G2's X/Y vs A/B buttons)
		// get components whose xrt_input the device never exposes, which RunFrame
		// then reports as missing every frame.
		const char *hand_subaction_path =
		    m_hand == XRT_HAND_LEFT ? "/user/hand/left" : "/user/hand/right";

		for (size_t i = 0; i < p->binding_count; i++) {
			struct binding_template *b = &p->bindings[i];

			if (b->subaction_path != NULL && strcmp(b->subaction_path, hand_subaction_path) != 0) {
				continue;
			}

			if (b->input != 0) {
				AddMonadoInput(b);
			}
			if (b->output != 0) {
				AddOutputControl(b->output, b->steamvr_path);
			}
		}
	}

	void
	PoseUpdateThreadFunction()
	{
		ovrd_log("Starting controller pose update thread for %s\n", m_xdev->str);

		while (m_poseUpdating) {
			//! @todo figure out the best pose update rate
			std::this_thread::sleep_for(std::chrono::milliseconds(1));

			if (m_unObjectId != vr::k_unTrackedDeviceIndexInvalid) {
				vr::VRServerDriverHost()->TrackedDevicePoseUpdated(m_unObjectId, GetPose(),
				                                                   sizeof(vr::DriverPose_t));
			}
		}

		ovrd_log("Stopping controller pose update thread for %s\n", m_xdev->str);
	}

	vr::EVRInitError
	Activate(vr::TrackedDeviceIndex_t unObjectId)
	{
		ovrd_log("Activating Controller SteamVR[%d]\n", unObjectId);

		if (!m_handed_controller) {
			//! @todo handle trackers etc
			ovrd_log("Unhandled: %s\n", m_xdev->str);
			return vr::VRInitError_Unknown;
		}

		m_unObjectId = unObjectId;

		if (this->m_xdev == NULL) {
			ovrd_log("Error: xdev NULL\n");
			return vr::VRInitError_Init_InterfaceNotFound;
		}

		// clang-format off


		m_ulPropertyContainer = vr::VRProperties()->TrackedDeviceToPropertyContainer(m_unObjectId);

		// return a constant that's not 0 (invalid) or 1 (reserved for Oculus)
		vr::VRProperties()->SetUint64Property(m_ulPropertyContainer, vr::Prop_CurrentUniverseId_Uint64, 2);
		vr::VRProperties()->SetInt32Property(m_ulPropertyContainer, vr::Prop_DeviceClass_Int32, vr::TrackedDeviceClass_Controller);

		if (m_hand == XRT_HAND_LEFT) {
			ovrd_log("Left Controller\n");
			vr::VRProperties()->SetInt32Property(m_ulPropertyContainer, vr::Prop_ControllerRoleHint_Int32, vr::TrackedControllerRole_LeftHand);
		} else if (m_hand == XRT_HAND_RIGHT) {
			ovrd_log("Right Controller\n");
			vr::VRProperties()->SetInt32Property(m_ulPropertyContainer, vr::Prop_ControllerRoleHint_Int32, vr::TrackedControllerRole_RightHand);
		}
		m_pose.poseIsValid = false;
		m_pose.deviceIsConnected = true;
		m_pose.result = vr::TrackingResult_Uninitialized;
		m_pose.willDriftInYaw = !m_xdev->supported.position_tracking;

		if (m_emulate_index_controller) {
			m_input_profile = std::string("{indexcontroller}/input/index_controller_profile.json");
			m_controller_type = "knuckles";
			if (m_hand == XRT_HAND_LEFT) {
				m_render_model = "{indexcontroller}valve_controller_knu_1_0_left";
			} else if (m_hand == XRT_HAND_RIGHT) {
				m_render_model = "{indexcontroller}valve_controller_knu_1_0_right";
			}
		} else {

			struct profile_template *p = get_profile_template(m_xdev->name);

			if (p == NULL) {
				ovrd_log("Monado device has unknown profile: %d\n", m_xdev->name);
				return vr::VRInitError_Unknown;
			}

			m_input_profile = std::string("{monado}/input/") + std::string(p->steamvr_input_profile_path);
			m_controller_type = p->steamvr_controller_type;
		}

		ovrd_log("Using input profile %s\n", m_input_profile.c_str());
		ovrd_log("Using render model%s\n", m_render_model);
		vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_InputProfilePath_String, m_input_profile.c_str());
		vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_RenderModelName_String, m_render_model);
		vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_ModelNumber_String, m_xdev->str);

		// clang-format on

		m_input_controls.clear();
		if (m_emulate_index_controller) {
			AddEmulatedIndexControls();
		} else {
			AddMonadoControls();
		}

		ovrd_log("Controller %d activated\n", m_unObjectId);

		m_poseUpdateThread = new std::thread(&CDeviceDriver_Monado_Controller::PoseUpdateThreadFunction, this);
		if (!m_poseUpdateThread) {
			ovrd_log("Unable to create pose updated thread for %s\n", m_xdev->str);
			return vr::VRInitError_Driver_Failed;
		}

		return vr::VRInitError_None;
	}

	void
	Deactivate()
	{
		ovrd_log("deactivate controller\n");
		m_poseUpdating = false;
		m_poseUpdateThread->join();
		m_unObjectId = vr::k_unTrackedDeviceIndexInvalid;
	}

	void
	EnterStandby()
	{
		ovrd_log("standby controller\n");
	}

	void *
	GetComponent(const char *pchComponentNameAndVersion)
	{
		// deprecated API
		// ovrd_log("get controller component %s\n",
		// pchComponentNameAndVersion);
		return NULL;
	}

	/** debug request from a client */
	void
	DebugRequest(const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize)
	{
		if (unResponseBufferSize >= 1)
			pchResponseBuffer[0] = 0;
	}

	vr::DriverPose_t
	GetPose()
	{
		// monado predicts pose "now", see xrt_device_get_tracked_pose
		m_pose.poseTimeOffset = 0;

		m_pose.poseIsValid = true;
		m_pose.result = vr::TrackingResult_Running_OK;
		m_pose.deviceIsConnected = true;

		enum xrt_input_name grip_name;

		//! @todo better method to find grip name
		if (m_xdev->name == XRT_DEVICE_VIVE_WAND) {
			grip_name = XRT_INPUT_VIVE_GRIP_POSE;
		} else if (m_xdev->name == XRT_DEVICE_INDEX_CONTROLLER) {
			grip_name = XRT_INPUT_INDEX_GRIP_POSE;
		} else if (m_xdev->name == XRT_DEVICE_FLIPVR) {
			grip_name = XRT_INPUT_FLIPVR_GRIP_POSE;
		} else if (m_xdev->name == XRT_DEVICE_PSMV) {
			grip_name = XRT_INPUT_PSMV_GRIP_POSE;
		} else if (m_xdev->name == XRT_DEVICE_DAYDREAM) {
			grip_name = XRT_INPUT_DAYDREAM_POSE;
		} else if (m_xdev->name == XRT_DEVICE_HYDRA) {
			grip_name = XRT_INPUT_HYDRA_GRIP_POSE;
		} else if (m_xdev->name == XRT_DEVICE_TOUCH_CONTROLLER) {
			grip_name = XRT_INPUT_TOUCH_GRIP_POSE;
		} else if (m_xdev->name == XRT_DEVICE_WMR_CONTROLLER) {
			grip_name = XRT_INPUT_WMR_GRIP_POSE;
		} else if (m_xdev->name == XRT_DEVICE_SAMSUNG_ODYSSEY_CONTROLLER) {
			grip_name = XRT_INPUT_ODYSSEY_CONTROLLER_GRIP_POSE;
		} else if (m_xdev->name == XRT_DEVICE_HP_REVERB_G2_CONTROLLER) {
			grip_name = XRT_INPUT_G2_CONTROLLER_GRIP_POSE;
		} else if (m_xdev->name == XRT_DEVICE_SIMPLE_CONTROLLER) {
			grip_name = XRT_INPUT_SIMPLE_GRIP_POSE;
		} else {
			ovrd_log("Unhandled device name %u\n", m_xdev->name);
			grip_name = XRT_INPUT_GENERIC_HEAD_POSE; // ???
		}

		timepoint_ns now_ns = os_monotonic_get_ns();

		struct xrt_space_relation rel;
		struct xrt_pose from_raw = {{0.f, 0.f, 0.f, 1.f}, {0.f, 0.f, 0.f}};
		uint64_t world_generation = 0;
		const bool paired_world = m_xdev->get_tracked_pose_with_presentation != nullptr;
		if (paired_world) {
			// Reporting-only body/reach logic needs the head in the same presentation frame.
			// Read the head's already-paired pose+guard, then convert to this tracking origin.
			struct xrt_pose presented_head, head_in_tracking;
			bool have_head = false;
			{
				std::lock_guard<std::mutex> lk(g_world_guard.lock);
				have_head = g_world_guard.have_prev;
				if (have_head) u_world_reanchor_apply(&g_world_guard.wr, &g_world_guard.prev_pose, &presented_head);
			}
			if (have_head) {
				struct xrt_pose inv_origin;
				math_pose_invert(&m_xdev->tracking_origin->initial_offset, &inv_origin);
				math_pose_transform(&inv_origin, &presented_head, &head_in_tracking);
			}
			m_xdev->get_tracked_pose_with_presentation(m_xdev, grip_name, now_ns, &rel,
			                                                &from_raw, &world_generation,
			                                                have_head ? &head_in_tracking : nullptr);
		} else {
			xrt_device_get_tracked_pose(m_xdev, grip_name, now_ns, &rel);
		}

		struct xrt_pose *offset = &m_xdev->tracking_origin->initial_offset;

		struct xrt_relation_chain chain = {};
		m_relation_chain_push_relation(&chain, &rel);
		m_relation_chain_push_pose_if_not_identity(&chain, offset);
		struct xrt_space_relation raw_in_origin;
		m_relation_chain_resolve(&chain, &raw_in_origin);

		struct xrt_pose presented;
		float delta7[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f};
		if (paired_world) {
			// Apply the correction paired with this exact raw snapshot, before the origin transform.
			// The relation chain rotates world velocities too; no glide derivative enters IMU rates.
			struct xrt_relation_chain presentation_chain = {};
			m_relation_chain_push_relation(&presentation_chain, &rel);
			m_relation_chain_push_pose_if_not_identity(&presentation_chain, &from_raw);
			m_relation_chain_push_pose_if_not_identity(&presentation_chain, offset);
			m_relation_chain_resolve(&presentation_chain, &rel);
			presented = rel.pose;

			// Telemetry dev1/2 delta7 is the FULL rigid transform in the final origin frame:
			// C_origin = origin * C_raw * inverse(origin); raw = inverse(C_origin)*presented.
			struct xrt_pose inv_origin, temp, origin_correction;
			math_pose_invert(offset, &inv_origin);
			math_pose_transform(offset, &from_raw, &temp);
			math_pose_transform(&temp, &inv_origin, &origin_correction);
			delta7[0]=origin_correction.position.x; delta7[1]=origin_correction.position.y;
			delta7[2]=origin_correction.position.z; delta7[3]=origin_correction.orientation.x;
			delta7[4]=origin_correction.orientation.y; delta7[5]=origin_correction.orientation.z;
			delta7[6]=origin_correction.orientation.w;
		} else {
			// Devices without the paired API keep their existing presentation behavior.
			rel = raw_in_origin;
			std::lock_guard<std::mutex> lk(g_world_guard.lock);
			u_world_reanchor_apply(&g_world_guard.wr, &rel.pose, &presented);
			world_guard_pack_delta(g_world_guard.wr, delta7);
		}

		apply_pose(&rel, &presented, &m_pose);

		if (g2_telem_enabled()) {
			const uint8_t dev_id =
			    m_xdev->device_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER    ? 1
			    : m_xdev->device_type == XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER ? 2
			                                                                   : 0;
			if (dev_id != 0) {
				const float pose7[7] = {presented.position.x,    presented.position.y,
				                        presented.position.z,    presented.orientation.x,
				                        presented.orientation.y, presented.orientation.z,
				                        presented.orientation.w};
				const float lv[3] = {rel.linear_velocity.x, rel.linear_velocity.y,
				                     rel.linear_velocity.z};
				const float av[3] = {rel.angular_velocity.x, rel.angular_velocity.y,
				                     rel.angular_velocity.z};
				g2_telem_getpose(dev_id, (uint64_t)now_ns, pose7, lv, av,
				                 (uint32_t)rel.relation_flags, delta7);
			}
		}

#ifdef DUMP_POSE_CONTROLLERS
		ovrd_log("get controller %d pose %f %f %f %f, %f %f %f\n", m_unObjectId, m_pose.qRotation.x,
		         m_pose.qRotation.y, m_pose.qRotation.z, m_pose.qRotation.w, m_pose.vecPosition[0],
		         m_pose.vecPosition[1], m_pose.vecPosition[2]);
#endif
		vr::HmdQuaternion_t identityquat{1, 0, 0, 0};
		m_pose.qWorldFromDriverRotation = identityquat;
		m_pose.qDriverFromHeadRotation = identityquat;
		m_pose.vecDriverFromHeadTranslation[0] = 0.f;
		m_pose.vecDriverFromHeadTranslation[1] = 0.f;
		m_pose.vecDriverFromHeadTranslation[2] = 0.f;

		return m_pose;
	}

	void
	RunFrame()
	{
		m_xdev->update_inputs(m_xdev);


		for (const auto &in : m_input_controls) {

			// ovrd_log("Update %d: %s\n", i,
			// m_controls[i].steamvr_control_path);

			enum xrt_input_name binding_name = in.monado_input_name;

			struct xrt_input *input = NULL;
			for (uint32_t ii = 0; ii < m_xdev->input_count; ii++) {
				if (m_xdev->inputs[ii].name == binding_name) {
					input = &m_xdev->inputs[ii];
					break;
				}
			}

			if (input == NULL) {
				ovrd_log("Input for %s not found!\n", in.steamvr_control_path);
				continue;
			}

			vr::VRInputComponentHandle_t handle = in.control_handle;

			if (in.monado_input_type == XRT_INPUT_TYPE_BOOLEAN) {
				bool state = input->value.boolean;
				vr::VRDriverInput()->UpdateBooleanComponent(handle, state, 0);
				// ovrd_log("Update %s: %d\n",
				// m_controls[i].steamvr_control_path, state);
				// U_LOG_D("Update %s: %d",
				//       m_controls[i].steamvr_control_path,
				//       state);
			}
			if (in.monado_input_type == XRT_INPUT_TYPE_VEC1_MINUS_ONE_TO_ONE ||
			    in.monado_input_type == XRT_INPUT_TYPE_VEC1_ZERO_TO_ONE ||
			    in.monado_input_type == XRT_INPUT_TYPE_VEC2_MINUS_ONE_TO_ONE) {

				float value;
				if (in.component.has_component && in.component.x) {
					value = input->value.vec2.x;
				} else if (in.component.has_component && in.component.y) {
					value = input->value.vec2.y;
				} else {
					value = input->value.vec1.x;
				}

				vr::VRDriverInput()->UpdateScalarComponent(handle, value, 0);
				// ovrd_log("Update %s: %f\n",
				// m_controls[i].steamvr_control_path,
				// state->x);
				// U_LOG_D("Update %s: %f",
				//       m_controls[i].steamvr_control_path,
				//       value);
			}
		}

		if (m_xdev->supported.hand_tracking && m_skeletal_input_control.control_handle) {
			constexpr const std::size_t sk_comp_size = 2;

			using ht_input_name_list = std::array<const enum xrt_input_name, sk_comp_size>;
			constexpr const std::array<const ht_input_name_list, sk_comp_size> ht_input_name_map = {
			    ht_input_name_list{
			        XRT_INPUT_HT_UNOBSTRUCTED_LEFT,
			        XRT_INPUT_HT_CONFORMING_LEFT,
			    },
			    ht_input_name_list{
			        XRT_INPUT_HT_UNOBSTRUCTED_RIGHT,
			        XRT_INPUT_HT_CONFORMING_RIGHT,
			    },
			};
			using bone_transform_list = std::array<vr::VRBoneTransform_t, OPENVR_BONE_COUNT>;
			using bone_transforms_list = std::vector<bone_transform_list>;

			bone_transforms_list bone_transforms;
			bone_transforms.reserve(sk_comp_size);
			const auto &ht_input_names = ht_input_name_map[m_hand];
			const timepoint_ns now_ns = os_monotonic_get_ns();

			for (size_t i = 0; i < ht_input_names.size(); ++i) {
				int64_t out_timestamp_ns;
				struct xrt_hand_joint_set joint_set_value = {};
				if (m_xdev->get_hand_tracking(m_xdev, ht_input_names[i], now_ns, &joint_set_value,
				                              &out_timestamp_ns) != XRT_SUCCESS) {
					continue;
				}
				auto &bts = bone_transforms.emplace_back();
				hand_joint_set_to_bone_transform(joint_set_value, bts.data(), m_hand);
				// hand_joint_set_to_bone_transforms(out_joint_set_value, bone_transforms);
			}

			if (bone_transforms.empty())
				return;
			if (bone_transforms.size() < sk_comp_size) {
				bone_transforms.push_back(bone_transforms.back());
			}
			assert(bone_transforms.size() >= sk_comp_size);

			vr::EVRInputError err = vr::VRDriverInput()->UpdateSkeletonComponent(
			    m_skeletal_input_control.control_handle, vr::VRSkeletalMotionRange_WithoutController,
			    bone_transforms[0].data(), OPENVR_BONE_COUNT);
			if (err != vr::VRInputError_None) {
				ovrd_log("error updating skeleton: %i ", err);
			}

			err = vr::VRDriverInput()->UpdateSkeletonComponent(
			    m_skeletal_input_control.control_handle, vr::VRSkeletalMotionRange_WithController,
			    bone_transforms[1].data(), OPENVR_BONE_COUNT);
			if (err != vr::VRInputError_None) {
				ovrd_log("error updating skeleton: %i ", err);
			}
		}
	}


	vr::VRControllerState_t
	GetControllerState()
	{
		// deprecated API
		vr::VRControllerState_t controllerstate{};
		return controllerstate;
	}

	bool
	TriggerHapticPulse(uint32_t unAxisId, uint16_t usPulseDurationMicroseconds)
	{
		// deprecated API
		return false;
	}

	std::string
	GetSerialNumber() const
	{
		ovrd_log("get controller serial number: %s\n", m_sSerialNumber);
		return m_sSerialNumber;
	}


	struct xrt_device *m_xdev;
	vr::DriverPose_t m_pose;
	vr::TrackedDeviceIndex_t m_unObjectId;
	vr::PropertyContainerHandle_t m_ulPropertyContainer;

	bool m_emulate_index_controller = false;

	std::vector<struct SteamVRDriverControlInput> m_input_controls;
	struct SteamVRDriverControlInput m_skeletal_input_control;

	std::vector<struct SteamVRDriverControlOutput> m_output_controls;

private:
	char m_sSerialNumber[XRT_DEVICE_NAME_LEN];
	char m_sModelNumber[MODELNUM_LEN];

	const char *m_controller_type = NULL;

	const char *m_render_model = NULL;
	enum xrt_hand m_hand;
	bool m_handed_controller;

	std::string m_input_profile;

	bool m_poseUpdating = true;
	std::thread *m_poseUpdateThread = NULL;
};

/*
 *
 * Device driver
 *
 */

class CDeviceDriver_Monado : public vr::ITrackedDeviceServerDriver, public vr::IVRDisplayComponent
{
public:
	CDeviceDriver_Monado(struct xrt_instance *xinst, struct xrt_device *xdev) : m_xdev(xdev)
	{
		// Photon latency from the device when it knows its panel;
		// OpenVR predicts poses this far past vsync, so an error here
		// is a per-frame motion-to-photon mispredict. 0.011 f is the
		// legacy fallback for devices that do not report it.
		uint64_t photons_ns = m_xdev->hmd->screens[0].vsync_to_photons_ns;
		m_flSecondsFromVsyncToPhotons = photons_ns != 0 ? (float)(photons_ns * 1e-9) : 0.011f;

		float ns = (float)m_xdev->hmd->screens->nominal_frame_interval_ns;
		m_flDisplayFrequency = 1.f / ns * 1000.f * 1000.f * 1000.f;
		ovrd_log("display frequency from device: %f\n", m_flDisplayFrequency);

		// steamvr can really misbehave when freq is inf or so
		if (m_flDisplayFrequency < 0 || m_flDisplayFrequency > 1000) {
			ovrd_log("Setting display frequency to 60 Hz!\n");
			m_flDisplayFrequency = 60.f;
		}

		//! @todo get ipd user setting from monado session
		float ipd_meters = 0.063f;
		struct xrt_vec3 ipd_vec = {ipd_meters, 0, 0};

		timepoint_ns now_ns = os_monotonic_get_ns();

		//! @todo more than 2 views
		struct xrt_space_relation head_relation;
		xrt_device_get_view_poses(xdev, &ipd_vec, now_ns, XRT_VIEW_TYPE_STEREO, 2, &head_relation, m_fovs,
		                          m_view_pose);

		//! @todo more versatile IPD calculation
		float actual_ipd = -m_view_pose[0].position.x + m_view_pose[1].position.x;

		m_flIPD = actual_ipd;

		ovrd_log("Seconds from Vsync to Photons: %f\n", m_flSecondsFromVsyncToPhotons);
		ovrd_log("Display Frequency: %f\n", m_flDisplayFrequency);
		ovrd_log("IPD: %f\n", m_flIPD);
	};
	// clang-format off
	virtual ~CDeviceDriver_Monado() {};

	// ITrackedDeviceServerDriver
	virtual vr::EVRInitError Activate(vr::TrackedDeviceIndex_t unObjectId);
	virtual void Deactivate();
	virtual void EnterStandby();
	virtual void *GetComponent(const char *pchComponentNameAndVersion);
	virtual void DebugRequest(const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize);
	virtual vr::DriverPose_t GetPose();

	// IVRDisplayComponent
	virtual void GetWindowBounds(int32_t *pnX, int32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight);
	virtual bool IsDisplayOnDesktop();
	virtual bool IsDisplayRealDisplay();
	virtual void GetRecommendedRenderTargetSize(uint32_t *pnWidth, uint32_t *pnHeight);
	virtual void GetEyeOutputViewport(vr::EVREye eEye, uint32_t *pnX, uint32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight);
	virtual void GetProjectionRaw(vr::EVREye eEye, float *pfLeft, float *pfRight, float *pfTop, float *pfBottom);
	virtual vr::DistortionCoordinates_t ComputeDistortion(vr::EVREye eEye, float fU, float fV);

	// clang-format on

private:
	struct xrt_device *m_xdev = NULL;

	// clang-format off

	vr::TrackedDeviceIndex_t m_trackedDeviceIndex = 0;
	vr::PropertyContainerHandle_t m_ulPropertyContainer = vr::k_ulInvalidPropertyContainer;

	float m_flSecondsFromVsyncToPhotons = -1;
	float m_flDisplayFrequency = -1;
	float m_flIPD = -1;

	struct xrt_fov m_fovs[2];
	struct xrt_pose m_view_pose[2];

	bool m_poseUpdating = true;
	std::thread *m_poseUpdateThread = NULL;
	virtual void PoseUpdateThreadFunction();

	void SetHiddenAreaMeshes();

	// clang-format on
};

static void
create_translation_rotation_matrix(struct xrt_pose *pose, struct vr::HmdMatrix34_t *res)
{
	struct xrt_vec3 t = pose->position;
	struct xrt_quat r = pose->orientation;
	res->m[0][0] = (1.0f - 2.0f * (r.y * r.y + r.z * r.z));
	res->m[1][0] = (r.x * r.y + r.z * r.w) * 2.0f;
	res->m[2][0] = (r.x * r.z - r.y * r.w) * 2.0f;
	res->m[0][1] = (r.x * r.y - r.z * r.w) * 2.0f;
	res->m[1][1] = (1.0f - 2.0f * (r.x * r.x + r.z * r.z));
	res->m[2][1] = (r.y * r.z + r.x * r.w) * 2.0f;
	res->m[0][2] = (r.x * r.z + r.y * r.w) * 2.0f;
	res->m[1][2] = (r.y * r.z - r.x * r.w) * 2.0f;
	res->m[2][2] = (1.0f - 2.0f * (r.x * r.x + r.y * r.y));
	res->m[0][3] = t.x;
	res->m[1][3] = t.y;
	res->m[2][3] = t.z;
}

void
CDeviceDriver_Monado::PoseUpdateThreadFunction()
{
	ovrd_log("Starting HMD pose update thread\n");

	while (m_poseUpdating) {
		//! @todo figure out the best pose update rate
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		vr::VRServerDriverHost()->TrackedDevicePoseUpdated(m_trackedDeviceIndex, GetPose(),
		                                                   sizeof(vr::DriverPose_t));
	}
	ovrd_log("Stopping HMD pose update thread\n");
}

void
CDeviceDriver_Monado::SetHiddenAreaMeshes()
{
	static const struct
	{
		enum xrt_visibility_mask_type xrt_type;
		vr::EHiddenAreaMeshType vr_type;
	} mesh_types[] = {
	    {XRT_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH, vr::k_eHiddenAreaMesh_Standard},
	    {XRT_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH, vr::k_eHiddenAreaMesh_Inverse},
	    {XRT_VISIBILITY_MASK_TYPE_LINE_LOOP, vr::k_eHiddenAreaMesh_LineLoop},
	};

	if (m_xdev->get_visibility_mask == NULL) {
		ovrd_log("Device provides no visibility mask, not setting hidden area meshes\n");
		return;
	}

	for (uint32_t eye = 0; eye < 2; eye++) {
		const struct xrt_fov &fov = m_xdev->hmd->distortion.fov[eye];
		const float tan_left = tanf(fov.angle_left);
		const float tan_right = tanf(fov.angle_right);
		const float tan_up = tanf(fov.angle_up);
		const float tan_down = tanf(fov.angle_down);

		for (const auto &mt : mesh_types) {
			struct xrt_visibility_mask *mask = NULL;
			xrt_result_t xret = xrt_device_get_visibility_mask(m_xdev, mt.xrt_type, eye, &mask);
			if (xret != XRT_SUCCESS || mask == NULL) {
				ovrd_log("Failed to get visibility mask type %d for eye %d: %d\n", mt.xrt_type, eye,
				         xret);
				continue;
			}

			const struct xrt_vec2 *verts = xrt_visibility_mask_get_vertices(mask);
			const uint32_t *indices = xrt_visibility_mask_get_indices(mask);

			// OpenVR wants per-eye render texture UVs (0,0 = upper left) and a
			// flat non-indexed vertex list: triangle list for the standard and
			// inverse types, vertex order for the line loop. The conversion is
			// the exact inverse of the UV->tangent mapping the mask was built
			// with, using the same per-eye FoV.
			std::vector<vr::HmdVector2_t> uvs(mask->index_count);
			for (uint32_t i = 0; i < mask->index_count; i++) {
				const struct xrt_vec2 &v = verts[indices[i]];
				uvs[i].v[0] = (v.x - tan_left) / (tan_right - tan_left);
				uvs[i].v[1] = (tan_up - v.y) / (tan_up - tan_down);
			}
			free(mask);

			if (mt.vr_type == vr::k_eHiddenAreaMesh_Standard) {
				double area = 0;
				for (size_t i = 0; i + 2 < uvs.size(); i += 3) {
					const double ax = uvs[i + 1].v[0] - uvs[i].v[0];
					const double ay = uvs[i + 1].v[1] - uvs[i].v[1];
					const double bx = uvs[i + 2].v[0] - uvs[i].v[0];
					const double by = uvs[i + 2].v[1] - uvs[i].v[1];
					area += fabs(ax * by - ay * bx) / 2;
				}
				ovrd_log("Hidden area mesh for eye %d: %u triangles covering %.1f%%\n", eye,
				         (uint32_t)(uvs.size() / 3), area * 100.0);
			}

			vr::ETrackedPropertyError err = vr::VRHiddenArea()->SetHiddenArea(
			    (vr::EVREye)eye, mt.vr_type, uvs.data(), (uint32_t)uvs.size());
			if (err != vr::TrackedProp_Success) {
				ovrd_log("Failed to set hidden area mesh type %d for eye %d: %d\n", mt.vr_type, eye,
				         err);
			}
		}
	}
}

vr::EVRInitError
CDeviceDriver_Monado::Activate(vr::TrackedDeviceIndex_t unObjectId)
{
	ovrd_log("Activate tracked device %u: %s\n", unObjectId, m_xdev->str);

	m_trackedDeviceIndex = unObjectId;

	// clang-format off

	m_ulPropertyContainer = vr::VRProperties()->TrackedDeviceToPropertyContainer(unObjectId);
	//! @todo: proper serial and model number

	vr::VRProperties()->SetInt32Property(m_ulPropertyContainer, vr::Prop_DeviceClass_Int32, vr::TrackedDeviceClass_HMD);
	vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_TrackingSystemName_String, "monado");
	vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_ManufacturerName_String, "Monado");
	vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_ModelNumber_String, m_xdev->str);
	vr::VRProperties()->SetFloatProperty(m_ulPropertyContainer, vr::Prop_UserIpdMeters_Float, m_flIPD);
	vr::VRProperties()->SetFloatProperty(m_ulPropertyContainer, vr::Prop_UserHeadToEyeDepthMeters_Float, 0.f);
	vr::VRProperties()->SetFloatProperty(m_ulPropertyContainer, vr::Prop_DisplayFrequency_Float, m_flDisplayFrequency);
	vr::VRProperties()->SetFloatProperty( m_ulPropertyContainer, vr::Prop_SecondsFromVsyncToPhotons_Float, m_flSecondsFromVsyncToPhotons);

	// return a constant that's not 0 (invalid) or 1 (reserved for Oculus)
	vr::VRProperties()->SetUint64Property( m_ulPropertyContainer, vr::Prop_CurrentUniverseId_Uint64, 2);

	// avoid "not fullscreen" warnings from vrmonitor
	//vr::VRProperties()->SetBoolProperty(m_ulPropertyContainer, vr::Prop_IsOnDesktop_Bool, false);

	// clang-format on


	//! @todo update when ipd changes
	vr::HmdMatrix34_t left;
	create_translation_rotation_matrix(&m_view_pose[0], &left);
	vr::HmdMatrix34_t right;
	create_translation_rotation_matrix(&m_view_pose[1], &right);

	vr::VRServerDriverHost()->SetDisplayEyeToHead(m_trackedDeviceIndex, left, right);

	SetHiddenAreaMeshes();

	m_poseUpdateThread = new std::thread(&CDeviceDriver_Monado::PoseUpdateThreadFunction, this);
	if (!m_poseUpdateThread) {
		ovrd_log("Unable to create pose updated thread for %s\n", m_xdev->str);
		return vr::VRInitError_Driver_Failed;
	}

	return vr::VRInitError_None;
}

void
CDeviceDriver_Monado::Deactivate()
{
	m_poseUpdating = false;
	m_poseUpdateThread->join();
	ovrd_log("Deactivate\n");
}

void
CDeviceDriver_Monado::EnterStandby()
{
	ovrd_log("Enter Standby\n");
}

void *
CDeviceDriver_Monado::GetComponent(const char *pchComponentNameAndVersion)
{
	// clang-format off
	if (strcmp(pchComponentNameAndVersion, vr::IVRDisplayComponent_Version) == 0) {
		return (vr::IVRDisplayComponent *)this;
	}
	// clang-format on

	return NULL;
}

void
CDeviceDriver_Monado::DebugRequest(const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize)
{
	//! @todo
}

static constexpr inline vr::HmdQuaternion_t
HmdQuaternion_Init(double w, double x, double y, double z)
{
	return vr::HmdQuaternion_t{
	    .w = w,
	    .x = x,
	    .y = y,
	    .z = z,
	};
}

vr::DriverPose_t
CDeviceDriver_Monado::GetPose()
{

	timepoint_ns now_ns = os_monotonic_get_ns();
	struct xrt_space_relation rel;
	xrt_device_get_tracked_pose(m_xdev, XRT_INPUT_GENERIC_HEAD_POSE, now_ns, &rel);

	// Motion signals for the re-anchor guard, from the raw device relation BEFORE the
	// origin-offset chain (magnitudes are rotation-invariant; the chain may drop unflagged
	// fields). On the SLAM path the velocity FIELDS carry the frame-corrected, IMU-derived
	// prediction velocities while the relation flags stay pose-only (wmr_hmd.c: forwarding head
	// velocities to the runtime is the L3 decision, not taken here) — exactly the design's
	// contamination rule: the envelope and caps key on the IMU-derived stream, never on a
	// raw-position finite difference the snap itself would inflate.
	const struct xrt_vec3 gv = rel.angular_velocity;
	const struct xrt_vec3 lv3 = rel.linear_velocity;
	const double gyro_dps =
	    sqrt((double)gv.x * gv.x + (double)gv.y * gv.y + (double)gv.z * gv.z) * (180.0 / M_PI);
	const double speed_mps = sqrt((double)lv3.x * lv3.x + (double)lv3.y * lv3.y + (double)lv3.z * lv3.z);

	struct xrt_pose *offset = &m_xdev->tracking_origin->initial_offset;

	struct xrt_relation_chain chain = {};
	m_relation_chain_push_relation(&chain, &rel);
	m_relation_chain_push_pose_if_not_identity(&chain, offset);
	m_relation_chain_resolve(&chain, &rel);

	// World re-anchor guard: detect a SLAM relocalization/reset step (innovation beyond the IMU
	// envelope), absorb only the excess into the shared delta, decay it (the glide). Zero-excess
	// pulls leave the delta untouched and present the raw pose bit-identically.
	struct xrt_pose presented = rel.pose;
	bool absorbed = false;
	double abs_ang_deg = 0.0;
	double abs_pos_m = 0.0;
	float delta7[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f};
	{
		std::lock_guard<std::mutex> lk(g_world_guard.lock);
		const enum xrt_space_relation_flags pose_valid = (enum xrt_space_relation_flags)(
		    XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_POSITION_VALID_BIT);
		if ((rel.relation_flags & pose_valid) == pose_valid) {
			if (g_world_guard.have_prev && now_ns > g_world_guard.prev_ns) {
				const double dt_s = (double)(now_ns - g_world_guard.prev_ns) * 1e-9;
				const double gyro_int_deg =
				    0.5 * (g_world_guard.prev_gyro_dps + gyro_dps) * dt_s;
				absorbed = u_world_reanchor_update(
				    &g_world_guard.wr, &u_world_reanchor_default_params, &g_world_guard.prev_pose,
				    &rel.pose, dt_s, gyro_int_deg, gyro_dps, speed_mps, &abs_ang_deg, &abs_pos_m);
			}
			g_world_guard.prev_pose = rel.pose;
			g_world_guard.prev_ns = now_ns;
			g_world_guard.prev_gyro_dps = gyro_dps;
			g_world_guard.have_prev = true;
		}
		u_world_reanchor_apply(&g_world_guard.wr, &rel.pose, &presented);
		world_guard_pack_delta(g_world_guard.wr, delta7);
	}
	if (absorbed && g2_telem_enabled()) {
		g2_telem_event(0, (uint64_t)now_ns, G2_TELEM_EV_WORLD_REANCHOR_ANG_DEG, (float)abs_ang_deg);
		g2_telem_event(0, (uint64_t)now_ns, G2_TELEM_EV_WORLD_REANCHOR_POS_M, (float)abs_pos_m);
	}

	vr::DriverPose_t t = {
	    // monado predicts pose "now", see xrt_device_get_tracked_pose
	    .poseTimeOffset = 0,
	    .qWorldFromDriverRotation = HmdQuaternion_Init(1, 0, 0, 0),
	    .vecWorldFromDriverTranslation = {0, 0, 0},

	    .qDriverFromHeadRotation = HmdQuaternion_Init(1, 0, 0, 0),
	    .vecDriverFromHeadTranslation = {0, 0, 0},

	    .vecPosition = {0, 0, 0},
	    .vecVelocity = {0, 0, 0},
	    .vecAcceleration = {0, 0, 0},

	    .qRotation = HmdQuaternion_Init(1, 0, 0, 0),

	    .vecAngularVelocity = {0, 0, 0},
	    .vecAngularAcceleration = {0, 0, 0},

	    .result = vr::TrackingResult_Running_OK,
	    .poseIsValid = (rel.relation_flags & XRT_SPACE_RELATION_ORIENTATION_VALID_BIT) != 0,
	    .willDriftInYaw = !m_xdev->supported.position_tracking,
	    //! @todo: Monado head model?
	    .shouldApplyHeadModel = !m_xdev->supported.position_tracking,
	    .deviceIsConnected = true,
	};
	apply_pose(&rel, &presented, &t);

	// S2's dev0 getpose tap: the presented head pose (the felt signal, post-guard — the
	// head_pose.bin telemetry tap stays PRE-guard by design so SLAM regressions can never hide
	// behind the glide) + the raw relation velocities/flags + the glide delta at this pull.
	if (g2_telem_enabled()) {
		const float pose7[7] = {presented.position.x,    presented.position.y,
		                        presented.position.z,    presented.orientation.x,
		                        presented.orientation.y, presented.orientation.z,
		                        presented.orientation.w};
		const float lv[3] = {rel.linear_velocity.x, rel.linear_velocity.y, rel.linear_velocity.z};
		const float av[3] = {rel.angular_velocity.x, rel.angular_velocity.y, rel.angular_velocity.z};
		g2_telem_getpose(0, (uint64_t)now_ns, pose7, lv, av, (uint32_t)rel.relation_flags, delta7);
	}

#ifdef DUMP_POSE
	ovrd_log("get hmd pose %f %f %f %f, %f %f %f\n", t.qRotation.x, t.qRotation.y, t.qRotation.z, t.qRotation.w,
	         t.vecPosition[0], t.vecPosition[1], t.vecPosition[2]);
#endif

	//! @todo
	// copy_vec3(&rel.angular_velocity, t.vecAngularVelocity);
	// copy_vec3(&rel.angular_acceleration, t.vecAngularAcceleration);

	// ovrd_log("Vel: %f %f %f\n", t.vecAngularVelocity[0],
	// t.vecAngularVelocity[1], t.vecAngularVelocity[2]); ovrd_log("Accel:
	// %f %f %f\n", t.vecAngularAcceleration[0],
	// t.vecAngularAcceleration[1], t.vecAngularAcceleration[2]);

	return t;
}

void
CDeviceDriver_Monado::GetWindowBounds(int32_t *pnX, int32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight)
{
	// offset in extended mode, e.g. to the right of a 1920x1080 monitor
	*pnX = 1920;
	*pnY = 0;

	*pnWidth = m_xdev->hmd->screens[0].w_pixels;
	*pnHeight = m_xdev->hmd->screens[0].h_pixels;
	;

	ovrd_log("Window Bounds: %dx%d\n", *pnWidth, *pnHeight);
}

bool
CDeviceDriver_Monado::IsDisplayOnDesktop()
{
	return false;
}

bool
CDeviceDriver_Monado::IsDisplayRealDisplay()
{
	return true;
}

void
CDeviceDriver_Monado::GetRecommendedRenderTargetSize(uint32_t *pnWidth, uint32_t *pnHeight)
{
	int scale = debug_get_num_option_scale_percentage();
	float fscale = (float)scale / 100.f;

	// OpenVR uses this size for each eye. The view display dimensions are in
	// client orientation; the screen dimensions cover the entire physical panel.
	const auto &left = m_xdev->hmd->views[vr::Eye_Left].display;
	const auto &right = m_xdev->hmd->views[vr::Eye_Right].display;
	*pnWidth = std::max(left.w_pixels, right.w_pixels) * fscale;
	*pnHeight = std::max(left.h_pixels, right.h_pixels) * fscale;

	ovrd_log("Render Target Size: %dx%d (%fx)\n", *pnWidth, *pnHeight, fscale);
}

void
CDeviceDriver_Monado::GetEyeOutputViewport(
    vr::EVREye eEye, uint32_t *pnX, uint32_t *pnY, uint32_t *pnWidth, uint32_t *pnHeight)
{
	*pnWidth = m_xdev->hmd->views[eEye].viewport.w_pixels;
	*pnHeight = m_xdev->hmd->views[eEye].viewport.h_pixels;

	*pnX = m_xdev->hmd->views[eEye].viewport.x_pixels;
	*pnY = m_xdev->hmd->views[eEye].viewport.y_pixels;

	ovrd_log("Output Viewport for eye %d: %dx%d offset %dx%d\n", eEye, *pnWidth, *pnHeight, *pnX, *pnY);
}

void
CDeviceDriver_Monado::GetProjectionRaw(vr::EVREye eEye, float *pfLeft, float *pfRight, float *pfTop, float *pfBottom)
{
	*pfLeft = tanf(m_xdev->hmd->distortion.fov[eEye].angle_left);
	*pfRight = tanf(m_xdev->hmd->distortion.fov[eEye].angle_right);
	*pfTop = tanf(-m_xdev->hmd->distortion.fov[eEye].angle_up);
	*pfBottom = tanf(-m_xdev->hmd->distortion.fov[eEye].angle_down);
	ovrd_log("Projection Raw: L%f R%f T%f B%f\n", *pfLeft, *pfRight, *pfTop, *pfBottom);
}

vr::DistortionCoordinates_t
CDeviceDriver_Monado::ComputeDistortion(vr::EVREye eEye, float fU, float fV)
{
	/** Used to return the post-distortion UVs for each color channel.
	 * UVs range from 0 to 1 with 0,0 in the upper left corner of the
	 * source render target. The 0,0 to 1,1 range covers a single eye. */

	struct xrt_vec2 *rot = m_xdev->hmd->views[eEye].rot.vecs;

	// multiply 2x2 rotation matrix with fU, fV scaled to [-1, 1]
	float U = rot[0].x * (fU * 2 - 1) + rot[0].y * (fV * 2 - 1);
	float V = rot[1].x * (fU * 2 - 1) + rot[1].y * (fV * 2 - 1);

	// scale U, V back to [0, 1]
	U = (U + 1) / 2;
	V = (V + 1) / 2;

	struct xrt_uv_triplet d;

	xrt_result_t xret = m_xdev->compute_distortion(m_xdev, eEye, U, V, &d);
	if (xret != XRT_SUCCESS) {
		ovrd_log("Failed to compute distortion for view %d at %f,%f! Got error %d\n", eEye, U, V, xret);

		vr::DistortionCoordinates_t coordinates;
		coordinates.rfBlue[0] = U;
		coordinates.rfBlue[1] = V;
		coordinates.rfGreen[0] = U;
		coordinates.rfGreen[1] = V;
		coordinates.rfRed[0] = U;
		coordinates.rfRed[1] = V;
		return coordinates;
	}

	vr::DistortionCoordinates_t coordinates;
	coordinates.rfRed[0] = d.r.x;
	coordinates.rfRed[1] = d.r.y;
	coordinates.rfGreen[0] = d.g.x;
	coordinates.rfGreen[1] = d.g.y;
	coordinates.rfBlue[0] = d.b.x;
	coordinates.rfBlue[1] = d.b.y;

	// ovrd_log("Computed distortion for view %d at %f,%f -> %f,%f | %f,%f |
	// %f,%f!\n", eEye, U, V, d.r.x, d.r.y, d.g.x, d.g.y, d.b.x, d.b.y);

	return coordinates;
}


/*
 *
 * Device driver server
 *
 */

class CServerDriver_Monado : public vr::IServerTrackedDeviceProvider
{
public:
	CServerDriver_Monado() : m_MonadoDeviceDriver(NULL) {}

	// clang-format off
	virtual ~CServerDriver_Monado() {};
	virtual vr::EVRInitError Init(vr::IVRDriverContext *pDriverContext);
	virtual void Cleanup();
	virtual const char *const * GetInterfaceVersions() { return vr::k_InterfaceVersions; }
	virtual void RunFrame();
	virtual bool ShouldBlockStandbyMode() { return false; }
	virtual void EnterStandby() {}
	virtual void LeaveStandby() {}
	virtual void HandleHapticEvent(vr::VREvent_t *event);
	// clang-format on

private:
	struct xrt_instance *m_xinst = NULL;
	struct xrt_system *m_xsys = NULL;
	struct xrt_system_devices *m_xsysd = NULL;
	struct xrt_space_overseer *m_xso = NULL;
	struct xrt_device *m_xhmd = NULL;

	CDeviceDriver_Monado *m_MonadoDeviceDriver = NULL;
	CDeviceDriver_Monado_Controller *m_left = NULL;
	CDeviceDriver_Monado_Controller *m_right = NULL;
};

CServerDriver_Monado g_serverDriverMonado;

#define NUM_XDEVS 16

vr::EVRInitError
CServerDriver_Monado::Init(vr::IVRDriverContext *pDriverContext)
{

	VR_INIT_SERVER_DRIVER_CONTEXT(pDriverContext);
	ovrd_log_init(vr::VRDriverLog());

	ovrd_log("Initializing Monado driver\n");

	//! @todo instance initialization is difficult to replicate

	xrt_result_t xret;
	xret = xrt_instance_create(NULL, &m_xinst);
	if (xret != XRT_SUCCESS) {
		ovrd_log("Failed to create instance\n");
		return vr::VRInitError_Init_HmdNotFound;
	}

	xret = xrt_instance_create_system(m_xinst, &m_xsys, &m_xsysd, &m_xso, NULL);
	if (xret < 0) {
		ovrd_log("Failed to create system devices\n");
		xrt_instance_destroy(&m_xinst);
		return vr::VRInitError_Init_HmdNotFound;
	}
	if (m_xsysd->static_roles.head == NULL) {
		ovrd_log("Didn't get a HMD device!\n");
		xrt_space_overseer_destroy(&m_xso);
		xrt_system_devices_destroy(&m_xsysd);
		xrt_system_destroy(&m_xsys);
		xrt_instance_destroy(&m_xinst);
		return vr::VRInitError_Init_HmdNotFound;
	}

	m_xhmd = m_xsysd->static_roles.head;

	ovrd_log("Selected HMD %s\n", m_xhmd->str);

	// Block until the HMD's native video mode is enumerated by X RandR before
	// returning from Init(). vrserver brings up vrcompositor immediately after
	// the driver loads, and on X11 vrcompositor scans RandR once to find a
	// direct-mode display matching the driver-reported HMD resolution. WMR
	// headsets only expose their native mode a few seconds after USB
	// activation, so without this wait vrcompositor scans too early and the
	// direct-mode lease fails.
	//
	// On Wayland, vrcompositor acquires the HMD via the wp_drm_lease_device_v1
	// protocol — not X RandR — so the wait is irrelevant and would burn the
	// full timeout for no benefit (the HMD is hidden from XWayland by the
	// compositor's non-desktop filtering). Skip the wait when a Wayland session
	// is detected.
	if (m_xhmd->hmd != NULL && m_xhmd->hmd->screens[0].w_pixels > 0) {
		uint32_t want_w = (uint32_t)m_xhmd->hmd->screens[0].w_pixels;
		uint32_t want_h = (uint32_t)m_xhmd->hmd->screens[0].h_pixels;
		const char *wl = getenv("WAYLAND_DISPLAY");
		if (wl != NULL && wl[0] != '\0') {
			ovrd_log("Wayland session detected (WAYLAND_DISPLAY=%s); skipping X RandR "
			         "wait for HMD mode %ux%u — vrcompositor leases via DRM.\n",
			         wl, want_w, want_h);
		} else {
			ovrd_log("Waiting for HMD display mode %ux%u to enumerate in X RandR...\n",
			         want_w, want_h);
			bool ready = ovrd_wait_for_hmd_randr_mode(want_w, want_h, 20000);
			ovrd_log("HMD display mode %ux%u in RandR: %s\n", want_w, want_h,
			         ready ? "present — vrcompositor can lease it"
			               : "not seen (timeout/no-X) — proceeding anyway");
		}
	}

	m_MonadoDeviceDriver = new CDeviceDriver_Monado(m_xinst, m_xhmd);
	//! @todo provide a serial number
	vr::VRServerDriverHost()->TrackedDeviceAdded(m_xhmd->str, vr::TrackedDeviceClass_HMD, m_MonadoDeviceDriver);

	xrt_system_roles system_roles = XRT_SYSTEM_ROLES_INIT;
	xrt_system_devices_get_roles(m_xsysd, &system_roles);

	struct xrt_device *left_xdev = nullptr;
	if (system_roles.left >= 0 && system_roles.left < XRT_SYSTEM_MAX_DEVICES) {
		left_xdev = m_xsysd->xdevs[system_roles.left];
	}
	struct xrt_device *right_xdev = nullptr;
	if (system_roles.right >= 0 && system_roles.right < XRT_SYSTEM_MAX_DEVICES) {
		right_xdev = m_xsysd->xdevs[system_roles.right];
	}

	// use steamvr room setup instead
	struct xrt_vec3 offset = {0, 0, 0};
	u_builder_setup_tracking_origins(m_xhmd, nullptr, left_xdev, right_xdev, nullptr, &offset);

	if (left_xdev) {
		m_left = new CDeviceDriver_Monado_Controller(m_xinst, left_xdev, XRT_HAND_LEFT);
		ovrd_log("Added left Controller: %s\n", left_xdev->str);
	}
	if (right_xdev) {
		m_right = new CDeviceDriver_Monado_Controller(m_xinst, right_xdev, XRT_HAND_RIGHT);
		ovrd_log("Added right Controller: %s\n", right_xdev->str);
	}

	return vr::VRInitError_None;
}

void
CServerDriver_Monado::Cleanup()
{
	if (m_MonadoDeviceDriver != NULL) {
		delete m_MonadoDeviceDriver;
		m_MonadoDeviceDriver = NULL;
	}

	xrt_space_overseer_destroy(&m_xso);
	xrt_system_devices_destroy(&m_xsysd);
	xrt_system_destroy(&m_xsys);
	m_xhmd = NULL;
	// m_left/m_right exist only if controller roles were assigned during Init(); WMR assigns
	// them a moment later, so these are commonly NULL even while the controllers are tracked.
	// Cleanup() lacked this guard -> NULL-deref core dump on every shutdown.
	if (m_left != NULL) {
		m_left->m_xdev = NULL;
	}
	if (m_right != NULL) {
		m_right->m_xdev = NULL;
	}

	if (m_xinst) {
		xrt_instance_destroy(&m_xinst);
	}
}

void
CServerDriver_Monado::HandleHapticEvent(vr::VREvent_t *event)
{
	float freq = event->data.hapticVibration.fFrequency;
	float amp = event->data.hapticVibration.fAmplitude;
	float duration = event->data.hapticVibration.fDurationSeconds;

	ovrd_log("Haptic vibration %fs %fHz %famp\n", duration, freq, amp);

	CDeviceDriver_Monado_Controller *controller = NULL;

	if (m_left && m_left->m_ulPropertyContainer == event->data.hapticVibration.containerHandle) {
		controller = m_left;
		ovrd_log("Haptic vibration left\n");
	} else if (m_right && m_right->m_ulPropertyContainer == event->data.hapticVibration.containerHandle) {
		controller = m_right;
		ovrd_log("Haptic vibration right\n");
	} else {
		ovrd_log("Haptic vibration ignored\n");
		return;
	}

	struct xrt_output_value out;
	out.vibration.amplitude = amp;
	if (duration > 0.00001) {
		out.vibration.duration_ns = (time_duration_ns)(duration * 1000.f * 1000.f * 1000.f);
	} else {
		out.vibration.duration_ns = XRT_MIN_HAPTIC_DURATION;
	}
	out.vibration.frequency = freq;

	if (controller->m_output_controls.empty()) {
		ovrd_log("Controller %s has no outputs\n", controller->m_xdev->str);
		return;
	}

	// TODO: controllers with more than 1 haptic motors
	SteamVRDriverControlOutput *control = &controller->m_output_controls.at(0);

	enum xrt_output_name name = control->monado_output_name;
	ovrd_log("Haptic vibration %s, %d\n", control->steamvr_control_path, name);
	controller->m_xdev->set_output(controller->m_xdev, name, &out);
}

void
CServerDriver_Monado::RunFrame()
{
	if (m_left) {
		m_left->RunFrame();
	}
	if (m_right) {
		m_right->RunFrame();
	}

	// https://github.com/ValveSoftware/openvr/issues/719#issuecomment-358038640
	struct vr::VREvent_t event;
	while (vr::VRServerDriverHost()->PollNextEvent(&event, sizeof(struct vr::VREvent_t))) {
		switch (event.eventType) {
		case vr::VREvent_Input_HapticVibration: HandleHapticEvent(&event); break;
		case vr::VREvent_PropertyChanged:
			// ovrd_log("Property changed\n");
			break;
		case vr::VREvent_TrackedDeviceActivated:
			ovrd_log("Device activated %d\n", event.trackedDeviceIndex);
			break;
		case vr::VREvent_TrackedDeviceUserInteractionStarted:
			ovrd_log("Device interaction started %d\n", event.trackedDeviceIndex);
			break;
		case vr::VREvent_IpdChanged: ovrd_log("ipd changed to %fm\n", event.data.ipd.ipdMeters); break;
		// This event currently spams the console, so is currently commented out. see
		// https://github.com/ValveSoftware/SteamVR-for-Linux/issues/307
		// case vr::VREvent_ActionBindingReloaded: ovrd_log("action binding reloaded\n"); break;
		case vr::VREvent_StatusUpdate: ovrd_log("EVRState: %d\n", event.data.status.statusState); break;

		case vr::VREvent_TrackedDeviceRoleChanged:
			// device roles are for legacy input
		case vr::VREvent_ChaperoneUniverseHasChanged:
		case vr::VREvent_ProcessQuit:
		case vr::VREvent_QuitAcknowledged:
		case vr::VREvent_ProcessDisconnected:
		case vr::VREvent_ProcessConnected:
		case vr::VREvent_DashboardActivated:
		case vr::VREvent_DashboardDeactivated:
		case vr::VREvent_Compositor_ChaperoneBoundsShown:
		case vr::VREvent_Compositor_ChaperoneBoundsHidden: break;

		default: ovrd_log("Unhandled Event: %d\n", event.eventType);
		}
	}
}


/*
 *
 * Watchdog code
 *
 */

class CWatchdogDriver_Monado : public vr::IVRWatchdogProvider
{
public:
	CWatchdogDriver_Monado()
	{
		ovrd_log("Created watchdog object\n");
		m_pWatchdogThread = nullptr;
	}

	// clang-format off
	virtual ~CWatchdogDriver_Monado() {};
	virtual vr::EVRInitError Init(vr::IVRDriverContext *pDriverContext);
	virtual void Cleanup();
	// clang-format on

private:
	std::thread *m_pWatchdogThread;
};

CWatchdogDriver_Monado g_watchdogDriverMonado;
bool g_bExiting = false;

void
WatchdogThreadFunction()
{
	while (!g_bExiting) {
#if defined(_WINDOWS)
		// on windows send the event when the Y key is pressed.
		if ((0x01 & GetAsyncKeyState('Y')) != 0) {
			// Y key was pressed.
			vr::VRWatchdogHost()->WatchdogWakeUp(vr::TrackedDeviceClass_HMD);
		}
		std::this_thread::sleep_for(std::chrono::microseconds(500));
#else
		ovrd_log("Watchdog wakeup\n");
		// for the other platforms, just send one every five seconds
		std::this_thread::sleep_for(std::chrono::seconds(1));
		vr::VRWatchdogHost()->WatchdogWakeUp(vr::ETrackedDeviceClass::TrackedDeviceClass_HMD);
#endif
	}

	ovrd_log("Watchdog exit\n");
}

vr::EVRInitError
CWatchdogDriver_Monado::Init(vr::IVRDriverContext *pDriverContext)
{
	VR_INIT_WATCHDOG_DRIVER_CONTEXT(pDriverContext);
	ovrd_log_init(vr::VRDriverLog());

	// Watchdog mode on Windows starts a thread that listens for the 'Y' key
	// on the keyboard to be pressed. A real driver should wait for a system
	// button event or something else from the the hardware that signals
	// that the VR system should start up.
	g_bExiting = false;

	ovrd_log("starting watchdog thread\n");

	m_pWatchdogThread = new std::thread(WatchdogThreadFunction);
	if (!m_pWatchdogThread) {
		ovrd_log("Unable to create watchdog thread\n");
		return vr::VRInitError_Driver_Failed;
	}

	return vr::VRInitError_None;
}

void
CWatchdogDriver_Monado::Cleanup()
{
	g_bExiting = true;
	if (m_pWatchdogThread) {
		m_pWatchdogThread->join();
		delete m_pWatchdogThread;
		m_pWatchdogThread = nullptr;
	}

	ovrd_log_cleanup();
}


/*
 *
 * 'Exported' functions.
 *
 */

void *
ovrd_hmd_driver_impl(const char *pInterfaceName, int *pReturnCode)
{
	// clang-format off
	if (0 == strcmp(vr::IServerTrackedDeviceProvider_Version, pInterfaceName)) {
		return &g_serverDriverMonado;
	}
	if (0 == strcmp(vr::IVRWatchdogProvider_Version, pInterfaceName)) {
		return &g_watchdogDriverMonado;
	}
	// clang-format on

	ovrd_log("Unimplemented interface: %s\n", pInterfaceName);

	if (pReturnCode) {
		*pReturnCode = vr::VRInitError_Init_InterfaceNotFound;
	}

	return NULL;
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
