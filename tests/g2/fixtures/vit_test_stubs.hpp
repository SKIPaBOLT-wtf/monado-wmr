// SPDX-License-Identifier: BSL-1.0
// Synthetic VIT transport/history stubs; no capture or calibration data.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>
using std::vector;
using xrt_space_relation_flags=int;
constexpr int XRT_SPACE_RELATION_BITMASK_ALL=63,XRT_SPACE_RELATION_ORIENTATION_VALID_BIT=1,XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT=8;
constexpr int VIT_SUCCESS=0;using vit_result_t=int;
struct xrt_vec3 {float x=0,y=0,z=0;};
struct xrt_quat {float x=0,y=0,z=0,w=1;};
struct xrt_pose {xrt_quat orientation;xrt_vec3 position;};
struct xrt_space_relation {xrt_pose pose;xrt_vec3 linear_velocity,angular_velocity;int relation_flags=0;};
#define XRT_SPACE_RELATION_ZERO xrt_space_relation{}
struct vit_pose_data_t {int64_t timestamp=0;float px=0,py=0,pz=0,ox=0,oy=0,oz=0,ow=1,vx=0,vy=0,vz=0;};
struct vit_pose_t {vit_pose_data_t data;vector<uint32_t> counts;int error_cam=-1;};
struct vit_pose_features {uint32_t count=0;};
struct xrt_pose_sample {int64_t timestamp;xrt_pose pose;};
struct Writer {vector<xrt_pose_sample> rows;void push(xrt_pose_sample p){rows.push_back(p);}};
struct Times {template<class T> void push(T) {} void push(std::pair<int64_t,vector<int>>) {}};
struct History {vector<std::pair<int64_t,xrt_space_relation>> rows;void push(xrt_space_relation r,int64_t t){rows.push_back({t,r});}bool get_latest(int64_t *t,xrt_space_relation*r){if(rows.empty())return false;*t=rows.back().first;*r=rows.back().second;return true;}};
struct Backend {vector<vit_pose_t> poses;size_t next=0;int resets=0,destroyed=0,feature_calls=0;};
struct VIT {
 int tracker_pop_pose(Backend*b,vit_pose_t**p){*p=b->next<b->poses.size()?&b->poses[b->next++]:nullptr;return 0;}
 int pose_get_data(vit_pose_t*p,vit_pose_data_t*d){*d=p->data;return 0;}
 int pose_get_features(const vit_pose_t*p,uint32_t i,vit_pose_features*f){++feature_calls;if(int(i)==p->error_cam)return -1;f->count=p->counts.at(i);return 0;}
 int tracker_reset(Backend*b){++b->resets;return 0;}
 void pose_destroy(vit_pose_t*){++destroyed;}
 int feature_calls=0,destroyed=0;
};
struct TrackerSlam {bool require_visual_observations=true,visual_observations_available=true;uint32_t cam_count=2;VIT vit;Backend backend;Backend*tracker=&backend;int slam_consec_bad=0,dbg_pred_counter=0,dbg_pred_every=1;History slam_rels;int lock_ff=0,gyro_ff=0;Writer raw;Writer*slam_traj_writer=&raw;Times times;Times*slam_times_writer=&times,*slam_features_writer=&times;struct{bool enabled=false;}features;struct Rec{int gt;}rec;Rec*euroc_recorder=&rec;};
#define SLAM_ERROR(...) ((void)0)
#define SLAM_TRACE(...) ((void)0)
#define SLAM_WARN(...) ((void)0)
#define SLAM_INFO(...) ((void)0)
constexpr int u_world_reanchor_default_params=0;
struct u_world_reanchor_step {double exc_ang_deg=0,dang_deg=0;};
static int derivatives=0;
float m_vec3_len(xrt_vec3 v){return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);}
xrt_vec3 m_vec3_mul_scalar(xrt_vec3 v,float a){return {v.x*a,v.y*a,v.z*a};}
double time_ns_to_s(int64_t t){return t*1e-9;}
void math_quat_finite_difference(xrt_quat*,xrt_quat*,double,xrt_vec3*out){++derivatives;*out={0,0,.1f};}
void os_mutex_lock(int*){}void os_mutex_unlock(int*){}
bool m_ff_vec3_f32_get(int,size_t,xrt_vec3*,uint64_t*){return false;}
bool u_world_reanchor_compute_excess(const int*,const xrt_pose*,const xrt_pose*,double,double,u_world_reanchor_step*){return false;}
void gt_ui_push(TrackerSlam&,int64_t,xrt_pose){}
void xrt_sink_push_pose(int,xrt_pose_sample*){}
vector<int64_t> timing_ui_push(TrackerSlam&,vit_pose_t*,int64_t){return {};}
vector<int> features_ui_push(TrackerSlam&,vit_pose_t*,int64_t){return {};}
