/* SPDX-License-Identifier: BSD-3-Clause
 * Local optional VIT extension. Include the unchanged vit_interface.h first.
 */
#pragma once
#include <stdint.h>

#if !defined(VIT_HEADER_VERSION_MAJOR)
#error "Include vit_interface.h before vit_g2_bias_v1.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VIT_G2_IMU_BIAS_V1_SIZE 64u
#define VIT_G2_IMU_BIAS_GYRO_VALID 1u
#define VIT_G2_IMU_BIAS_ACCEL_VALID 2u

/* Fixed v1 layout, copied from the SAME immutable state as pose_get_data.
 * Bias is in calibrated IMU/body axes at timestamp_ns, BEFORE body-to-world
 * rotation. Subtract it AFTER the backend's static calibration:
 *   gyro = M_full * raw - static_offset - gyro_bias_rad_s
 *   accel = M_lower * raw - static_offset - accel_bias_m_s2
 * M_lower includes the diagonal and lower triangle (Basalt accel convention).
 * These are learned residual biases, not a second copy of factory offsets.
 * Availability is not visual tracking confidence or evidence of stillness.
 */
typedef struct vit_g2_imu_bias_v1 {
  uint32_t struct_size;
  uint32_t flags;
  int64_t timestamp_ns;
  double gyro_bias_rad_s[3];
  double accel_bias_m_s2[3];
} vit_g2_imu_bias_v1;

/* Optional symbol: an unmodified backend does not export it. A valid backend-
 * owned live pose handle is required, exactly as for standard VIT pose getters.
 * Valid foreign implementation handles may be rejected; arbitrary pointers are
 * outside the C API contract and cannot be made safe by runtime type checks.
 * If out_size < 64 or out is NULL, return INVALID_VALUE without writing out.
 * Otherwise initialize the 64-byte output to unavailable (size=64, flags=0).
 * On SUCCESS the caller must check size, known flags, exact matching pose
 * timestamp and finite values before use. Unknown flag bits are unsupported.
 */
typedef vit_result_t (*PFN_vit_g2_pose_get_imu_bias_v1)(const vit_pose_t *pose,
                                                     uint32_t out_size,
                                                     vit_g2_imu_bias_v1 *out);

vit_result_t vit_g2_pose_get_imu_bias_v1(const vit_pose_t *pose,
                                       uint32_t out_size,
                                       vit_g2_imu_bias_v1 *out);

#ifdef __cplusplus
}
static_assert(sizeof(vit_g2_imu_bias_v1) == VIT_G2_IMU_BIAS_V1_SIZE, "v1 ABI size");
#else
_Static_assert(sizeof(vit_g2_imu_bias_v1) == VIT_G2_IMU_BIAS_V1_SIZE, "v1 ABI size");
#endif
