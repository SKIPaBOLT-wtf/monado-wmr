// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief World re-anchor glide implementation (see u_world_reanchor.h).
 *
 * Faithful C port of the validated reference simulator
 * results/b2-resnap-design-20260704/sim/sim_core.py — the update order (absorb ->
 * sanity cap -> rate-capped decay -> sub-perceptual snap), the quaternion
 * conventions (xyzw, world/left composition, hemisphere-safe step extraction)
 * and every formula match it exactly; tests_world_reanchor pins the port against
 * the simulator's outputs on the recorded head streams.
 */

#include "u_world_reanchor.h"

#include "math/m_mathinclude.h"

#include <string.h>

const struct u_world_reanchor_params u_world_reanchor_default_params = {
    .tau_s = 0.10,
    .omega0_dps = 60.0,
    .v0_mps = 1.0,
    .beta_w = 2.0,
    .beta_v = 1.0,
    .ori_floor_deg = 0.35,
    .snap_ang_deg = 0.05,
    .snap_pos_m = 0.002,
    .cap_ang_deg = 90.0,
    .cap_pos_m = 1.5,
    .env_gyro_gain = 1.5,
    .env_speed_mps = 2.5,
    .env_pos_floor_m = 0.008,
};

#define WR_EPS 1e-12
#define RAD2DEG (180.0 / M_PI)
#define DEG2RAD (M_PI / 180.0)

/* Hamilton product a (x) b, xyzw: apply b then a (world/left composition). */
static void
qmul(const double a[4], const double b[4], double out[4])
{
	const double x = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
	const double y = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0];
	const double z = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3];
	const double w = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
	out[0] = x;
	out[1] = y;
	out[2] = z;
	out[3] = w;
}

static void
quat_axis_angle(const double axis[3], double ang_rad, double out[4])
{
	const double s = sin(0.5 * ang_rad);
	out[0] = axis[0] * s;
	out[1] = axis[1] * s;
	out[2] = axis[2] * s;
	out[3] = cos(0.5 * ang_rad);
}

/* Rotation angle of a quaternion (deg), hemisphere-safe (arccos of |w|, as the simulator). */
static double
quat_angle_deg(const double q[4])
{
	double w = fabs(q[3]);
	if (w > 1.0) {
		w = 1.0;
	}
	return RAD2DEG * 2.0 * acos(w);
}

/* Same axis, new angle (slerp from identity); identity if the axis is degenerate. */
static void
quat_scale_angle(const double q_in[4], double new_ang_deg, double out[4])
{
	double q[4] = {q_in[0], q_in[1], q_in[2], q_in[3]};
	if (q[3] < 0.0) {
		q[0] = -q[0];
		q[1] = -q[1];
		q[2] = -q[2];
		q[3] = -q[3];
	}
	const double s = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
	if (s < WR_EPS) {
		out[0] = out[1] = out[2] = 0.0;
		out[3] = 1.0;
		return;
	}
	const double axis[3] = {q[0] / s, q[1] / s, q[2] / s};
	quat_axis_angle(axis, DEG2RAD * new_ang_deg, out);
}

void
u_world_reanchor_init(struct u_world_reanchor *wr)
{
	wr->dq[0] = wr->dq[1] = wr->dq[2] = 0.0;
	wr->dq[3] = 1.0;
	wr->dp[0] = wr->dp[1] = wr->dp[2] = 0.0;
	wr->active = false;
}

bool
u_world_reanchor_compute_excess(const struct u_world_reanchor_params *prm,
                                const struct xrt_pose *prev,
                                const struct xrt_pose *next,
                                double dt_s,
                                double gyro_int_deg,
                                struct u_world_reanchor_step *out_step)
{
	struct u_world_reanchor_step st;
	memset(&st, 0, sizeof(st));

	/* Hemisphere-align the pair, then world-frame step rotation R_step = q_next * q_prev^-1. */
	double q0[4] = {prev->orientation.x, prev->orientation.y, prev->orientation.z, prev->orientation.w};
	double q1[4] = {next->orientation.x, next->orientation.y, next->orientation.z, next->orientation.w};
	if (q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3] < 0.0) {
		for (int i = 0; i < 4; i++) {
			q1[i] = -q1[i];
		}
	}
	const double q0inv[4] = {-q0[0], -q0[1], -q0[2], q0[3]};
	double step[4];
	qmul(q1, q0inv, step);
	if (step[3] < 0.0) {
		for (int i = 0; i < 4; i++) {
			step[i] = -step[i];
		}
	}
	const double sn = sqrt(step[0] * step[0] + step[1] * step[1] + step[2] * step[2]);
	st.dang_deg = RAD2DEG * 2.0 * atan2(sn, step[3]);
	if (sn > WR_EPS) {
		st.axis[0] = step[0] / sn;
		st.axis[1] = step[1] / sn;
		st.axis[2] = step[2] / sn;
	} else {
		st.axis[0] = 1.0;
		st.axis[1] = 0.0;
		st.axis[2] = 0.0;
	}

	const double dvec[3] = {
	    (double)next->position.x - (double)prev->position.x,
	    (double)next->position.y - (double)prev->position.y,
	    (double)next->position.z - (double)prev->position.z,
	};
	st.dnorm_m = sqrt(dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]);
	if (st.dnorm_m > WR_EPS) {
		st.unit_dvec[0] = dvec[0] / st.dnorm_m;
		st.unit_dvec[1] = dvec[1] / st.dnorm_m;
		st.unit_dvec[2] = dvec[2] / st.dnorm_m;
	}

	const double ori_env = prm->env_gyro_gain * gyro_int_deg + prm->ori_floor_deg;
	double pos_env = prm->env_speed_mps * dt_s;
	if (pos_env < prm->env_pos_floor_m) {
		pos_env = prm->env_pos_floor_m;
	}
	st.exc_ang_deg = st.dang_deg > ori_env ? st.dang_deg - ori_env : 0.0;
	st.exc_pos_m = st.dnorm_m > pos_env ? st.dnorm_m - pos_env : 0.0;

	*out_step = st;
	return st.exc_ang_deg > 0.0 || st.exc_pos_m > 0.0;
}

void
u_world_reanchor_step_to_world_delta(const struct u_world_reanchor_step *step,
                                     const struct xrt_vec3 *pivot,
                                     struct xrt_pose *out_delta)
{
	double dq[4];
	quat_axis_angle(step->axis, DEG2RAD * step->exc_ang_deg, dq);
	out_delta->orientation.x = (float)dq[0];
	out_delta->orientation.y = (float)dq[1];
	out_delta->orientation.z = (float)dq[2];
	out_delta->orientation.w = (float)dq[3];

	/* Rotate the pivot to place the pivot-anchored rotation into the x' = R x + t form,
	 * then the translation excess measured at the pivot: t = pivot + d_exc - R * pivot. */
	const double p[3] = {pivot->x, pivot->y, pivot->z};
	const double x = dq[0], y = dq[1], z = dq[2], w = dq[3];
	double rp[3];
	/* R(q) * p via the quaternion sandwich, expanded. */
	const double tx = 2.0 * (y * p[2] - z * p[1]);
	const double ty = 2.0 * (z * p[0] - x * p[2]);
	const double tz = 2.0 * (x * p[1] - y * p[0]);
	rp[0] = p[0] + w * tx + (y * tz - z * ty);
	rp[1] = p[1] + w * ty + (z * tx - x * tz);
	rp[2] = p[2] + w * tz + (x * ty - y * tx);

	out_delta->position.x = (float)(p[0] + step->unit_dvec[0] * step->exc_pos_m - rp[0]);
	out_delta->position.y = (float)(p[1] + step->unit_dvec[1] * step->exc_pos_m - rp[1]);
	out_delta->position.z = (float)(p[2] + step->unit_dvec[2] * step->exc_pos_m - rp[2]);
}

bool
u_world_reanchor_update(struct u_world_reanchor *wr,
                        const struct u_world_reanchor_params *prm,
                        const struct xrt_pose *raw_prev,
                        const struct xrt_pose *raw_new,
                        double dt_s,
                        double gyro_int_deg,
                        double gyro_dps,
                        double speed_mps,
                        double *out_abs_ang_deg,
                        double *out_abs_pos_m)
{
	struct u_world_reanchor_step st;
	u_world_reanchor_compute_excess(prm, raw_prev, raw_new, dt_s, gyro_int_deg, &st);

	bool absorbed = false;
	if (st.exc_ang_deg > 0.0) {
		/* Compose the step's excess fraction, inverted, into dq. The excess rotation
		 * R_exc lives in the RAW world frame (step = q1*q0^-1) while dq maps raw ->
		 * presented (presented_q = dq*raw_q), so exact continuity at the event
		 * (dq_new*q1 == dq_old*q0) requires RIGHT-composition: dq_new = dq_old*R_exc^-1.
		 * Left-composition only agrees when dq is identity or the axes commute — a
		 * second re-anchor mid-glide on a skew axis would present the commutator as a
		 * spurious jump (audit 2026-07-10 R2#1). */
		double inv_exc[4];
		quat_axis_angle(st.axis, -DEG2RAD * st.exc_ang_deg, inv_exc);
		double dq_new[4];
		qmul(wr->dq, inv_exc, dq_new);
		memcpy(wr->dq, dq_new, sizeof(dq_new));
		absorbed = true;
	}
	if (st.exc_pos_m > 0.0) {
		wr->dp[0] -= st.unit_dvec[0] * st.exc_pos_m;
		wr->dp[1] -= st.unit_dvec[1] * st.exc_pos_m;
		wr->dp[2] -= st.unit_dvec[2] * st.exc_pos_m;
		absorbed = true;
	}
	if (out_abs_ang_deg != NULL) {
		*out_abs_ang_deg = st.exc_ang_deg;
	}
	if (out_abs_pos_m != NULL) {
		*out_abs_pos_m = st.exc_pos_m;
	}

	if (wr->active || absorbed) {
		double ang = quat_angle_deg(wr->dq);
		double dpn = sqrt(wr->dp[0] * wr->dp[0] + wr->dp[1] * wr->dp[1] + wr->dp[2] * wr->dp[2]);

		/* Sanity cap: beyond it the remainder passes through instantly. */
		if (ang > prm->cap_ang_deg) {
			quat_scale_angle(wr->dq, prm->cap_ang_deg, wr->dq);
			ang = prm->cap_ang_deg;
		}
		if (dpn > prm->cap_pos_m) {
			const double k = prm->cap_pos_m / dpn;
			wr->dp[0] *= k;
			wr->dp[1] *= k;
			wr->dp[2] *= k;
			dpn = prm->cap_pos_m;
		}

		/* Rate-capped exponential decay; caps are motion-adaptive (perceptual masking). */
		const double w_max = prm->omega0_dps + prm->beta_w * gyro_dps;
		const double v_max = prm->v0_mps + prm->beta_v * speed_mps;
		const double ar = fmin(w_max, ang / prm->tau_s);
		const double pr = fmin(v_max, dpn / prm->tau_s);
		double new_ang = ang - ar * dt_s;
		double new_dpn = dpn - pr * dt_s;
		if (new_ang < 0.0) {
			new_ang = 0.0;
		}
		if (new_dpn < 0.0) {
			new_dpn = 0.0;
		}

		if (new_ang < prm->snap_ang_deg && new_dpn < prm->snap_pos_m) {
			u_world_reanchor_init(wr);
		} else {
			if (ang > 0.0 && new_ang < ang) {
				quat_scale_angle(wr->dq, new_ang, wr->dq);
			}
			if (dpn > 0.0 && new_dpn < dpn) {
				const double k = new_dpn / dpn;
				wr->dp[0] *= k;
				wr->dp[1] *= k;
				wr->dp[2] *= k;
			}
			wr->active = new_ang > 0.0 || new_dpn > 0.0;
		}
	}

	return absorbed;
}

void
u_world_reanchor_apply(const struct u_world_reanchor *wr, const struct xrt_pose *raw, struct xrt_pose *out_presented)
{
	if (!wr->active) {
		/* Exact copy: clean frames are bit-identical to the raw stream. */
		*out_presented = *raw;
		return;
	}
	const double q[4] = {raw->orientation.x, raw->orientation.y, raw->orientation.z, raw->orientation.w};
	double pres_q[4];
	qmul(wr->dq, q, pres_q);
	out_presented->orientation.x = (float)pres_q[0];
	out_presented->orientation.y = (float)pres_q[1];
	out_presented->orientation.z = (float)pres_q[2];
	out_presented->orientation.w = (float)pres_q[3];
	out_presented->position.x = (float)((double)raw->position.x + wr->dp[0]);
	out_presented->position.y = (float)((double)raw->position.y + wr->dp[1]);
	out_presented->position.z = (float)((double)raw->position.z + wr->dp[2]);
}

void
u_world_reanchor_get_magnitude(const struct u_world_reanchor *wr, double *out_ang_deg, double *out_pos_m)
{
	if (out_ang_deg != NULL) {
		*out_ang_deg = quat_angle_deg(wr->dq);
	}
	if (out_pos_m != NULL) {
		*out_pos_m = sqrt(wr->dp[0] * wr->dp[0] + wr->dp[1] * wr->dp[1] + wr->dp[2] * wr->dp[2]);
	}
}


static void
wr_rotate(const double q[4], const double v[3], double out[3])
{
    const double tx = 2.0 * (q[1] * v[2] - q[2] * v[1]);
    const double ty = 2.0 * (q[2] * v[0] - q[0] * v[2]);
    const double tz = 2.0 * (q[0] * v[1] - q[1] * v[0]);
    out[0] = v[0] + q[3] * tx + q[1] * tz - q[2] * ty;
    out[1] = v[1] + q[3] * ty + q[2] * tx - q[0] * tz;
    out[2] = v[2] + q[3] * tz + q[0] * ty - q[1] * tx;
}

/* Exact solution of y'=-min(cap,y/tau), with the existing glide parameters. */
static double
wr_decay(double magnitude, double cap, double tau, double elapsed)
{
    if (magnitude <= 0.0 || elapsed <= 0.0) return magnitude;
    const double linear_time = fmax(0.0, (magnitude - cap * tau) / cap);
    if (elapsed < linear_time) return magnitude - cap * elapsed;
    return fmin(magnitude, cap * tau) * exp(-(elapsed - linear_time) / tau);
}

void
u_world_reanchor_compensation_init(struct u_world_reanchor_compensation *s)
{
    memset(s, 0, sizeof(*s));
    s->dq[3] = 1.0;
}

static void
wr_compensation_eval(const struct u_world_reanchor_compensation *s, int64_t when_ns,
                     double q[4], double t[3])
{
    q[0] = q[1] = q[2] = t[0] = t[1] = t[2] = 0.0;
    q[3] = 1.0;
    if (!s->active) return;
    const struct u_world_reanchor_params *prm = &u_world_reanchor_default_params;
    const double elapsed = when_ns > s->anchor_ns ? (double)(when_ns - s->anchor_ns) * 1e-9 : 0.0;
    const double ang = quat_angle_deg(s->dq);
    const double pos = sqrt(s->dp[0]*s->dp[0]+s->dp[1]*s->dp[1]+s->dp[2]*s->dp[2]);
    const double a = wr_decay(ang, s->angular_cap_dps, prm->tau_s, elapsed);
    const double p = wr_decay(pos, s->linear_cap_mps, prm->tau_s, elapsed);
    if (a < prm->snap_ang_deg && p < prm->snap_pos_m) return;
    quat_scale_angle(s->dq, a, q);
    double rp[3]; wr_rotate(q, s->pivot, rp);
    const double k = pos > 0.0 ? p / pos : 0.0;
    for (int i=0;i<3;i++) t[i] = s->pivot[i] + k*s->dp[i] - rp[i];
}

void
u_world_reanchor_compensation_evaluate(const struct u_world_reanchor_compensation *s,
                                      int64_t when_ns, struct xrt_pose *out)
{
    double q[4], t[3]; wr_compensation_eval(s, when_ns, q, t);
    out->orientation = (struct xrt_quat){(float)q[0],(float)q[1],(float)q[2],(float)q[3]};
    out->position = (struct xrt_vec3){(float)t[0],(float)t[1],(float)t[2]};
}

bool
u_world_reanchor_compensation_rebase(struct u_world_reanchor_compensation *s,
                                    const struct xrt_pose *delta, const struct xrt_vec3 *pivot,
                                    int64_t when_ns, double gyro_dps, double speed_mps)
{
    if (!delta || !pivot || (s->generation != 0 && when_ns <= s->anchor_ns) || when_ns <= 0 ||
        !isfinite(gyro_dps) || !isfinite(speed_mps) || gyro_dps < 0 || speed_mps < 0) return false;
    double d[4] = {delta->orientation.x,delta->orientation.y,delta->orientation.z,delta->orientation.w};
    const double v[3] = {delta->position.x,delta->position.y,delta->position.z};
    const double h[3] = {pivot->x,pivot->y,pivot->z};
    double n=0; for(int i=0;i<4;i++){if(!isfinite(d[i]))return false;n+=d[i]*d[i];}
    for(int i=0;i<3;i++)if(!isfinite(v[i])||!isfinite(h[i]))return false;
    if(n < 1e-12)return false;
    n=sqrt(n);for(int i=0;i<4;i++)d[i]/=n;
    double old_q[4],old_t[3];wr_compensation_eval(s,when_ns,old_q,old_t);
    const double inv_q[4]={-d[0],-d[1],-d[2],d[3]};
    double q[4],inv_v[3],t[3];qmul(old_q,inv_q,q);wr_rotate(inv_q,v,inv_v);
    for(int i=0;i<3;i++)inv_v[i]=-inv_v[i];
    wr_rotate(old_q,inv_v,t);for(int i=0;i<3;i++)t[i]+=old_t[i];
    double rh[3];wr_rotate(q,h,rh);
    const struct u_world_reanchor_params *prm=&u_world_reanchor_default_params;
    double dp[3];for(int i=0;i<3;i++)dp[i]=rh[i]+t[i]-h[i];
    const double ang=quat_angle_deg(q);
    const double pos=sqrt(dp[0]*dp[0]+dp[1]*dp[1]+dp[2]*dp[2]);
    if(ang > prm->cap_ang_deg)quat_scale_angle(q,prm->cap_ang_deg,q);
    const double k=pos > prm->cap_pos_m ? prm->cap_pos_m/pos : 1.0;
    memcpy(s->dq,q,sizeof(q));memcpy(s->pivot,h,sizeof(h));
    for(int i=0;i<3;i++)s->dp[i]=dp[i]*k;
    s->anchor_ns=when_ns;s->generation++;
    s->angular_cap_dps=prm->omega0_dps+prm->beta_w*gyro_dps;
    s->linear_cap_mps=prm->v0_mps+prm->beta_v*speed_mps;
    s->active=ang > 0.0 || pos > 0.0;
    return true;
}
