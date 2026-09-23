#!/usr/bin/env python3
# SPDX-License-Identifier: BSL-1.0
"""Production mask/reanchor extraction, actual pose math, no device/runtime calls."""
from pathlib import Path
import hashlib,json,re,subprocess,tempfile,atexit,argparse
repo=Path(__file__).resolve().parents[2]
fixtures=Path(__file__).resolve().parent/'fixtures'
ap=argparse.ArgumentParser()
ap.add_argument('--build',type=Path,default=repo/'build/g2-stability')
args=ap.parse_args()
_work=tempfile.TemporaryDirectory(prefix='g2-mask-test-')
atexit.register(_work.cleanup)
root=Path(_work.name)
src=repo/'src/xrt/tracking/constellation/t_constellation_tracking.c'
old=fixtures/'mask_baseline.c'
s=src.read_text();prior=old.read_text()
def extract(text,marker,start=0):
 a=text.index(marker,start);b=text.index('{',a);depth=0
 for i in range(b,len(text)):
  depth+=(text[i]=='{')-(text[i]=='}')
  if not depth:return text[a:i+1]
 raise ValueError(marker)
pre=r'''
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "xrt/xrt_defines.h"
#include "xrt/xrt_tracking.h"
#include "math/m_api.h"
static int checks,groups,callbacks,pushes;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"check line%d: %s\n",__LINE__,#x);exit(42);}}while(0)
#define CONSTELLATION_MAX_CAMERAS 4
#define CONSTELLATION_MAX_DEVICES 2
#define U_TIME_1MS_IN_NS 1000000
#define ROI_MAX_OPTICAL_AGE_MS 150.0
#define MASK_HALO_MARGIN_PX 20.0f
#define MASK_AREA_CAP_FRACTION .20f
#define MASK_SIGMA_MIN_DEPTH_M .05f
#define PRIOR_GATE_SIGMA 3.0
#define G2_TELEM_MASK_ENABLED 1
#define G2_TELEM_MASK_FROM_PREDICTION 2
#define G2_TELEM_MASK_FROM_LAST_SEEN 4
#define G2_TELEM_MASK_AREA_CAPPED 8
struct pose_rect{double left,top,right,bottom;};
static bool pose_rect_has_area(struct pose_rect*r){return r->left!=r->right&&r->top!=r->bottom;}
struct t_constellation_led_model{int dummy;};
struct camera_model{struct{float fx,fy,cx,cy;}calib;int width,height;};
struct constellation_tracker_camera_state{int slam_tracking_index;struct camera_model camera_model;};
struct tracking_sample_frame{struct xrt_pose P_cam_world;};
struct tracking_sample_device_state{int dev_index;float prior_pos_std_m;bool prior_position_tracked,have_last_seen_pose;struct xrt_pose P_world_obj_prior,last_seen_pose;struct t_constellation_led_model*led_model;};
struct t_constellation_tracked_device_connection{void*xdev;int id;};
struct constellation_tracker_device{bool have_last_seen_pose;uint64_t last_seen_pose_ts;struct xrt_pose last_seen_pose;struct t_constellation_tracked_device_connection*connection;};
struct constellation_tracking_sample{int n_views,n_devices;uint64_t timestamp;struct tracking_sample_frame views[4];struct tracking_sample_device_state devices[2];};
struct t_constellation_tracker{int tracked_device_lock,num_devices;struct constellation_tracker_device devices[2];struct constellation_tracker_camera_state cam[4];struct xrt_device_masks_sample controller_masks_sample;struct xrt_device_masks_sink*controller_masks_sink;};
static void os_mutex_lock(int*m){CHECK(*m==0);*m=1;}static void os_mutex_unlock(int*m){CHECK(*m==1);*m=0;}
static bool g2_telem_enabled(void){return false;}static uint8_t telem_device_id(void*p){(void)p;return 0;}
static void g2_telem_mask(uint8_t c,uint64_t t,uint8_t d,uint8_t flags,const float*r,float sigma,float age){(void)c;(void)t;(void)d;(void)flags;(void)r;(void)sigma;(void)age;}
static struct t_constellation_tracker*active;
static bool require_callback_lock=true;
static struct xrt_pose notified_delta;static uint64_t notified_time,notified_publication;
static void constellation_tracked_device_connection_notify_world_reanchor(struct t_constellation_tracked_device_connection*c,int64_t t,const struct xrt_pose*d,const struct xrt_vec3*p,int64_t pub,double gyro,double speed){CHECK(c);CHECK(active->tracked_device_lock==(require_callback_lock?1:0));CHECK(p&&gyro==7&&speed==.3);notified_delta=*d;notified_time=t;notified_publication=pub;callbacks++;}
static void sink(struct xrt_device_masks_sink*s,struct xrt_device_masks_sample*m){(void)s;CHECK(active->tracked_device_lock==1);CHECK(m);pushes++;}
// Projection leaf is a deterministic 8-corner fixture, using actual production quaternion math.
static void pose_metrics_get_device_bounds(const struct xrt_pose*p,struct t_constellation_led_model*l,struct camera_model*c,struct pose_rect*r,void*a,void*b){
 (void)l;(void)a;(void)b;*r=(struct pose_rect){1e9,1e9,-1e9,-1e9};int n=0;
 for(int bits=0;bits<8;bits++){struct xrt_vec3 x={(bits&1)?-.03f:.03f,(bits&2)?-.02f:.02f,(bits&4)?-.01f:.01f},v;math_quat_rotate_vec3(&p->orientation,&x,&v);v.x+=p->position.x;v.y+=p->position.y;v.z+=p->position.z;if(v.z<=0)continue;double u=c->calib.fx*v.x/v.z+c->calib.cx,w=c->calib.fy*v.y/v.z+c->calib.cy;r->left=fmin(r->left,u);r->right=fmax(r->right,u);r->top=fmin(r->top,w);r->bottom=fmax(r->bottom,w);n++;}
 if(!n)*r=(struct pose_rect){0};else{r->left=fmax(0,r->left);r->top=fmax(0,r->top);r->right=fmin(c->width-1,r->right);r->bottom=fmin(c->height-1,r->bottom);}
}
static const struct xrt_pose P_YZ_FLIP={{1,0,0,0},{0,0,0}};
'''
funcs='\n'.join(extract(s,marker) for marker in ['static void\npose_flip_YZ(','static bool\nmask_union_device_rect(','static bool\ncontroller_mask_observation_fresh(','static void\npush_controller_masks(','static void\nconstellation_reanchor_devices('])
oldpush=extract(prior,'static void\npush_controller_masks(').replace('push_controller_masks(','baseline_push_controller_masks(',1)
loop=extract(prior,'for (int i = 0; i < ct->num_devices; i++)',prior.index('// One publication clock and pivot'))
oldreanchor='''static void baseline_reanchor(struct t_constellation_tracker*ct,int64_t frame_mono_ns,const struct xrt_pose*delta_ptr,const struct xrt_vec3*new_raw_pivot,int64_t publication_ns,double gyro_dps,double speed_mps){struct xrt_pose delta=*delta_ptr;struct {uint64_t timestamp;}frame={frame_mono_ns};void *unused=0;(void)unused;__typeof__(frame)*xf=&frame;struct xrt_pose transformed_pivot={.position=*new_raw_pivot};'''+loop+'}\n'
main=r'''
static struct t_constellation_tracker ct;static struct constellation_tracking_sample sample;static struct xrt_device_masks_sink output;static struct t_constellation_led_model model;static struct t_constellation_tracked_device_connection conn[2];
static struct xrt_pose pose(float x,float y,float z,float angle,float ax,float ay,float az){float norm=sqrtf(ax*ax+ay*ay+az*az),h=angle/2;return(struct xrt_pose){{sinf(h)*ax/norm,sinf(h)*ay/norm,sinf(h)*az/norm,cosf(h)},{x,y,z}};}
static void init(void){memset(&ct,0,sizeof(ct));memset(&sample,0,sizeof(sample));active=&ct;ct.num_devices=1;ct.controller_masks_sink=&output;output.push_device_masks=sink;ct.cam[0].camera_model=(struct camera_model){{400,410,320,240},640,480};sample.n_views=sample.n_devices=1;sample.timestamp=10000000000ULL;sample.views[0].P_cam_world=pose(0,0,0,0,1,0,0);sample.devices[0]=(struct tracking_sample_device_state){.dev_index=0,.prior_pos_std_m=.002,.prior_position_tracked=false,.have_last_seen_pose=true,.P_world_obj_prior=pose(.05,-.04,1.2,.3,1,2,3),.last_seen_pose=pose(.05,-.04,1.2,.3,1,2,3),.led_model=&model};ct.devices[0]=(struct constellation_tracker_device){.have_last_seen_pose=true,.last_seen_pose_ts=sample.timestamp,.last_seen_pose=sample.devices[0].last_seen_pose,.connection=&conn[0]};require_callback_lock=true;}
static bool runmask(bool old){ct.tracked_device_lock=1;if(old)baseline_push_controller_masks(&ct,&sample);else push_controller_masks(&ct,&sample);ct.tracked_device_lock=0;return ct.controller_masks_sample.views[0].devices[0].enabled;}
static struct xrt_rect_f32 rect(void){return ct.controller_masks_sample.views[0].devices[0].rect;}
static double err(struct xrt_rect_f32 a,struct xrt_rect_f32 b){return fmax(fmax(fabs(a.x-b.x),fabs(a.y-b.y)),fmax(fabs(a.w-b.w),fabs(a.h-b.h)));}
static void copycache(void){sample.devices[0].have_last_seen_pose=ct.devices[0].have_last_seen_pose;sample.devices[0].last_seen_pose=ct.devices[0].last_seen_pose;sample.devices[0].P_world_obj_prior=ct.devices[0].last_seen_pose;}
int main(void){
 CHECK(controller_mask_observation_fresh(1000000000,true,850000000));CHECK(!controller_mask_observation_fresh(1000000000,true,849999999));CHECK(!controller_mask_observation_fresh(1000000000,true,1000000001));CHECK(!controller_mask_observation_fresh(1000000000,false,1000000000));groups++;
 init();ct.devices[0].last_seen_pose_ts=sample.timestamp-150000000;CHECK(runmask(false));ct.devices[0].last_seen_pose_ts--;CHECK(!runmask(false));CHECK(runmask(true));groups++;
 init();ct.devices[0].last_seen_pose_ts=sample.timestamp+1;CHECK(!runmask(false));CHECK(runmask(true));groups++;
 init();sample.devices[0].have_last_seen_pose=false;ct.devices[0].have_last_seen_pose=false;CHECK(!runmask(false));CHECK(runmask(true));groups++;
 init();sample.devices[0].prior_pos_std_m=-1;CHECK(runmask(false));ct.devices[0].last_seen_pose_ts=sample.timestamp-150000001;CHECK(!runmask(false));groups++;
 init();sample.devices[0].have_last_seen_pose=false;ct.devices[0].have_last_seen_pose=false;sample.devices[0].prior_position_tracked=true;CHECK(runmask(false));struct xrt_rect_f32 tracked=rect();CHECK(runmask(true));CHECK(err(tracked,rect())==0);groups++;
 init();CHECK(runmask(false));struct xrt_rect_f32 fresh=rect();CHECK(runmask(true));CHECK(err(fresh,rect())==0);groups++;
 init();sample.devices[0].prior_position_tracked=true;sample.devices[0].prior_pos_std_m=10;CHECK(!runmask(false));CHECK(!runmask(true));groups++;
 init();sample.devices[0].last_seen_pose.position.z=-2;sample.devices[0].P_world_obj_prior.position.z=-2;CHECK(!runmask(false));CHECK(!runmask(true));groups++;
 // Full rigid reanchor invariance: nonidentity camera, device and world delta.
 init();sample.devices[0].prior_pos_std_m=-1;struct xrt_pose camera=pose(.2,-.4,.3,.55,1,2,-1),local=pose(.08,-.04,1.4,.4,2,-1,1);math_pose_transform(&camera,&local,&ct.devices[0].last_seen_pose);copycache();math_pose_invert(&camera,&sample.views[0].P_cam_world);CHECK(runmask(false));struct xrt_rect_f32 before=rect();
 struct xrt_pose oldcache=ct.devices[0].last_seen_pose;uint64_t stamp=ct.devices[0].last_seen_pose_ts;struct xrt_pose delta=pose(.3,-.2,.15,.35,1,2,3),dcv,newcamera;pose_flip_YZ(&delta,&dcv);math_pose_transform(&dcv,&camera,&newcamera);struct xrt_vec3 pivot={.1,.2,.3};int priorcallbacks=callbacks;
 constellation_reanchor_devices(&ct,sample.timestamp,&delta,&pivot,sample.timestamp+1000,7,.3);CHECK(callbacks==priorcallbacks+1&&ct.tracked_device_lock==0);CHECK(ct.devices[0].last_seen_pose_ts==stamp);CHECK(memcmp(&notified_delta,&delta,sizeof(delta))==0&&notified_time==sample.timestamp&&notified_publication==sample.timestamp+1000);copycache();math_pose_invert(&newcamera,&sample.views[0].P_cam_world);CHECK(runmask(false));double fixed_error=err(before,rect());CHECK(fixed_error<.001);groups++;
 // The old real notify-only loop does not transform the frontend cache.
 ct.devices[0].last_seen_pose=oldcache;copycache();require_callback_lock=false;baseline_reanchor(&ct,sample.timestamp,&delta,&pivot,sample.timestamp+1000,7,.3);CHECK(memcmp(&oldcache,&ct.devices[0].last_seen_pose,sizeof(oldcache))==0);CHECK(runmask(true));double old_error=err(before,rect());CHECK(old_error>10);groups++;
 // Cache timestamp stays old across rebase and still expires; disconnected cache is also world-correct.
 require_callback_lock=true;ct.devices[0].connection=NULL;constellation_reanchor_devices(&ct,sample.timestamp,&delta,&pivot,sample.timestamp+2000,7,.3);CHECK(ct.devices[0].last_seen_pose_ts==stamp);copycache();sample.timestamp=stamp+150000001;CHECK(!runmask(false));groups++;
 // Each device transformed once; a device without cache keeps its uninitialized storage untouched.
 init();ct.num_devices=2;ct.devices[1].connection=&conn[1];ct.devices[1].have_last_seen_pose=false;ct.devices[1].last_seen_pose=pose(9,8,7,.2,1,2,3);struct xrt_pose ignored=ct.devices[1].last_seen_pose;priorcallbacks=callbacks;constellation_reanchor_devices(&ct,sample.timestamp,&delta,&pivot,sample.timestamp+1000,7,.3);CHECK(callbacks==priorcallbacks+2&&memcmp(&ignored,&ct.devices[1].last_seen_pose,sizeof(ignored))==0);groups++;
 printf("{\"passed\":true,\"groups\":%d,\"assertions\":%d,\"full_se3_mask_error_px\":%.9f,\"baseline_unreanchored_error_px\":%.9f,\"stale_mask_baseline_counterexample\":true}\n",groups,checks,fixed_error,old_error);
}
'''
code=pre+funcs+oldpush+oldreanchor+main
cpp=root/'independent_mask_safety.c';exe=root/'independent_mask_safety';cpp.write_text(code)
mathlib=args.build.resolve()/'steamvr-monado/bin/linux64/driver_monado.so';source=repo
cmd=['nice','-n','19','cc','-std=gnu11','-O0','-g',str(cpp),'-I'+str(source/'src/xrt/include'),'-I'+str(source/'src/xrt/auxiliary'),'-I'+str(args.build.resolve()/'src/xrt/include'),str(mathlib),'-lm','-Wl,-rpath,'+str(mathlib.parent),'-o',str(exe)]
subprocess.run(cmd,check=True)
run=subprocess.run(['nice','-n','19',str(exe)],capture_output=True,text=True,timeout=15)
if run.returncode:raise RuntimeError(run.stderr)
assert src.read_text()==s,'Source changed during test'
out={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'baseline_source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'math_library_sha256':hashlib.sha256(mathlib.read_bytes()).hexdigest(),'result':json.loads(run.stdout),'scope':'Exact production mask union/push, freshness, YZ conversion, reanchor helper and baseline mask/notify-loop bodies. Actual frozen Monado pose math. Synthetic pinhole8-corner leaf bounds, tracked-lock assertions and callback/sink recording. No runtime/hardware calls. Baseline controls demonstrate stale untracked mask and frame mismatch; candidate preserves freshness/capping/full-SE3 invariance.'}
(root/'independent-mask-safety.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
