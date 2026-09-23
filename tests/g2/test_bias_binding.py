#!/usr/bin/env python3
# SPDX-License-Identifier: BSL-1.0
"""Extract actual reader/drain/reset bodies; synthetic ABI and history, no VR APIs."""
from pathlib import Path
import hashlib,json,subprocess,tempfile,atexit
repo=Path(__file__).resolve().parents[2]
fixtures=Path(__file__).resolve().parent/'fixtures'
_work=tempfile.TemporaryDirectory(prefix='g2-bias-test-')
atexit.register(_work.cleanup)
root=Path(_work.name)
src=repo/'src/xrt/auxiliary/tracking/t_tracker_slam.cpp'
source=src.read_text()
def extract(marker):
    a=source.index(marker);b=source.index('{',a);depth=0
    for i in range(b,len(source)):
        depth+=(source[i]=='{')-(source[i]=='}')
        if not depth:return source[a:i+1]
    raise ValueError(marker)
pre=(fixtures/'vit_test_stubs.hpp').read_text()
pre=pre.replace('#include <cassert>','#include <cassert>\n#include <cfloat>\n#include <cstdlib>\n#include <mutex>')
pre=pre.replace('using std::vector;','using std::vector; using std::unique_lock; using timepoint_ns=int64_t;\nstatic int checks=0;\n#undef assert\n#define assert(x) do {checks++;if(!(x)){std::cerr<<"check line "<<__LINE__<<": "<<#x<<"\\n";std::exit(42);}}while(0)')
sample=extract('struct gyro_bias_sample\n')+';\n'
pre=pre.replace('struct vit_pose_data_t',sample+'struct BiasWriter {vector<gyro_bias_sample> rows;void push(gyro_bias_sample s){rows.push_back(s);}};\nstruct vit_pose_data_t',1)
pre=pre.replace('int error_cam=-1;','int error_cam=-1;')
header=repo/'src/external/vit_includes/vit/vit_g2_bias_v1.h'
pre=pre.replace('struct vit_pose_features', '#define VIT_HEADER_VERSION_MAJOR 2\n#include "'+str(header.resolve())+'"\nstruct vit_pose_features',1)
pre=pre.replace('void push(xrt_space_relation r,int64_t t){rows.push_back({t,r});}', 'bool push(xrt_space_relation r,int64_t t){if(!rows.empty()&&t<=rows.back().first)return false;rows.push_back({t,r});return true;}')
pre=pre.replace('struct VIT {','struct VIT {\n PFN_vit_g2_pose_get_imu_bias_v1 g2_pose_get_imu_bias_v1=nullptr;int reset_result=0;')
pre=pre.replace('int tracker_reset(Backend*b){++b->resets;return 0;}','int tracker_reset(Backend*b){++b->resets;return reset_result;}')
pre=pre.replace('struct TrackerSlam {','struct TrackerSlam {bool use_pose_gyro_bias=true,have_prediction_imu_calib=true;gyro_bias_sample latest_gyro_bias{};BiasWriter biases;BiasWriter*gyro_bias_writer=&biases;std::mutex pose_mutex;')
reader=extract('static gyro_bias_sample\nread_pose_gyro_bias(')
drain=extract('static bool\nflush_poses_locked(')
count=extract('static int64_t\nvisual_observation_count(')
reset_lambda=extract('u_var_button_cb reset_state_cb = [](void *t_ptr)')
reset='static void production_reset(TrackerSlam &t){ using u_var_button_cb=void(*)(void*); '+reset_lambda+';reset_state_cb(&t);}\n'
main=r'''
static int calls=0;static vit_g2_imu_bias_v1 response{};static int response_code=0;static bool use_actual_ts=true;
static int callback(const vit_pose_t *p,uint32_t size,vit_g2_imu_bias_v1 *out){assert(p);assert(size==64);calls++;*out=response;if(use_actual_ts)out->timestamp_ns=p->data.timestamp;return response_code;}
static void good(){response={};response.struct_size=64;response.flags=3;response.gyro_bias_rad_s[0]=.01;response.gyro_bias_rad_s[1]=-.02;response.gyro_bias_rad_s[2]=.03;response_code=0;use_actual_ts=true;}
static vit_pose_t pose(int64_t ts,float p=2,uint32_t a=5,uint32_t b=4){vit_pose_t x;x.data.timestamp=ts;x.data.px=p;x.counts={a,b};return x;}
static void init(TrackerSlam&t){good();t.vit.g2_pose_get_imu_bias_v1=callback;}
static bool same(gyro_bias_sample a,gyro_bias_sample b){return a.ts==b.ts&&a.valid==b.valid&&a.bias.x==b.bias.x&&a.bias.y==b.bias.y&&a.bias.z==b.bias.z;}
static void one(TrackerSlam&t,vit_pose_t p){t.backend.poses.push_back(p);assert(flush_poses_locked(t));}
static void bound(TrackerSlam&t,int64_t ts,bool valid){assert(!t.slam_rels.rows.empty());assert(t.slam_rels.rows.back().first==ts);assert(t.latest_gyro_bias.ts==ts&&t.latest_gyro_bias.valid==valid);}
static void unavailable(TrackerSlam&t,const vit_pose_t&p){auto before=t.latest_gyro_bias;auto x=read_pose_gyro_bias(t,&p,p.data.timestamp);assert(x.ts==p.data.timestamp&&!x.valid);assert(x.bias.x==0&&x.bias.y==0&&x.bias.z==0);assert(same(before,t.latest_gyro_bias));}
int main(){int groups=0;
 {TrackerSlam t;init(t);auto p=pose(100);t.latest_gyro_bias={7,true,{1,2,3}};
  for(unsigned flags:{1u,3u}){good();response.flags=flags;auto x=read_pose_gyro_bias(t,&p,100);assert(x.valid&&x.ts==100);assert(x.bias.x==float(.01)&&x.bias.y==float(-.02)&&x.bias.z==float(.03));assert(t.latest_gyro_bias.ts==7);}groups++;
  for(int which=0;which<3;which++){good();t.use_pose_gyro_bias=true;t.have_prediction_imu_calib=true;t.vit.g2_pose_get_imu_bias_v1=callback;if(which==0)t.use_pose_gyro_bias=false;if(which==1)t.have_prediction_imu_calib=false;if(which==2)t.vit.g2_pose_get_imu_bias_v1=nullptr;int before=calls;unavailable(t,p);assert(calls==before);}groups++;
  init(t);t.use_pose_gyro_bias=true;t.have_prediction_imu_calib=true;
  for(unsigned size:{0u,63u,65u,72u}){good();response.struct_size=size;unavailable(t,p);}groups++;
  for(unsigned flags:{0u,2u,4u,5u,7u,0xffffffffu}){good();response.flags=flags;unavailable(t,p);}groups++;
  for(int64_t ts:{0LL,99LL,101LL}){good();use_actual_ts=false;response.timestamp_ns=ts;unavailable(t,p);}groups++;
  good();response_code=-1;unavailable(t,p);groups++;
  for(int axis=0;axis<3;axis++)for(double v:{double(NAN),double(INFINITY),double(-INFINITY),double(FLT_MAX)*2}){good();response.gyro_bias_rad_s[axis]=v;unavailable(t,p);}groups++;
  good();response.accel_bias_m_s2[0]=NAN;assert(read_pose_gyro_bias(t,&p,100).valid);groups++; // gyro-only consumer
 }
 {TrackerSlam t;init(t);one(t,pose(100));bound(t,100,true);assert(t.biases.rows.size()==1&&t.vit.destroyed==1);auto original=t.latest_gyro_bias;int prior_calls=calls;
  response.gyro_bias_rad_s[0]=.8;one(t,pose(100));assert(same(original,t.latest_gyro_bias)&&calls==prior_calls&&t.biases.rows.size()==1);one(t,pose(99));assert(same(original,t.latest_gyro_bias)&&calls==prior_calls);groups++;
  one(t,pose(101,2001));assert(same(original,t.latest_gyro_bias)&&calls==prior_calls&&t.slam_rels.rows.size()==1);groups++;
  one(t,pose(102,NAN));assert(same(original,t.latest_gyro_bias)&&calls==prior_calls);groups++;
  good();response_code=-1;one(t,pose(103));bound(t,103,false);assert(t.biases.rows.size()==2);groups++;
  good();one(t,pose(104));bound(t,104,true);assert(t.latest_gyro_bias.bias.x==float(.01));groups++;
  prior_calls=calls;one(t,pose(105,0,0,0));bound(t,105,false);assert(t.slam_rels.rows.back().second.relation_flags==0&&calls==prior_calls&&t.biases.rows.back().ts==105&&!t.biases.rows.back().valid);groups++;
  one(t,pose(106));bound(t,106,true);assert((t.slam_rels.rows.back().second.relation_flags&8)==0);groups++;
  prior_calls=calls;one(t,pose(107,2001,0,0));bound(t,107,false);assert(t.slam_rels.rows.back().second.relation_flags==0&&calls==prior_calls&&t.biases.rows.back().ts==107);groups++;
 }
 {TrackerSlam t;init(t);t.dbg_pred_every=3;one(t,pose(100));auto original=t.latest_gyro_bias;int prior_calls=calls;one(t,pose(101));one(t,pose(102));assert(same(original,t.latest_gyro_bias)&&calls==prior_calls&&t.biases.rows.size()==1);one(t,pose(103));bound(t,103,true);groups++;
  one(t,pose(104,0,0,0));bound(t,104,false);assert(t.biases.rows.size()==3);groups++; // invalid overrides debug skip
 }
 for(bool fail:{false,true}){TrackerSlam t;init(t);one(t,pose(100));t.vit.reset_result=fail?-1:0;production_reset(t);assert(!t.latest_gyro_bias.valid&&t.latest_gyro_bias.ts==0&&t.backend.resets==1);assert(t.slam_rels.rows.back().first==100);one(t,pose(101));bound(t,101,true);groups++;}
 for(bool fail:{false,true}){TrackerSlam t;init(t);one(t,pose(100));t.vit.reset_result=fail?-1:0;auto original=t.latest_gyro_bias;for(int i=0;i<29;i++){one(t,pose(101+i,2001));assert(same(original,t.latest_gyro_bias));}one(t,pose(130,2001));assert(!t.latest_gyro_bias.valid&&t.latest_gyro_bias.ts==0&&t.backend.resets==1);assert(t.slam_rels.rows.back().first==100);one(t,pose(131));bound(t,131,true);groups++;}
 for(int which=0;which<3;which++){TrackerSlam t;init(t);if(which==0)t.use_pose_gyro_bias=false;if(which==1)t.have_prediction_imu_calib=false;if(which==2)t.vit.g2_pose_get_imu_bias_v1=nullptr;int before=calls;one(t,pose(100));bound(t,100,false);assert(calls==before&&t.slam_rels.rows.back().second.relation_flags!=0);groups++;}
 std::cout<<"{\"passed\":true,\"groups\":"<<groups<<",\"assertions\":"<<checks<<"}\n";
}
'''
basecode=pre+count+'\n'+reader+'\n'+drain+'\n'+reset+main
def run(code,stem):
    cpp=root/(stem+'.cpp');exe=root/stem;cpp.write_text(code)
    subprocess.run(['nice','-n','19','g++','-std=c++17','-O0','-g',str(cpp),'-o',str(exe)],check=True)
    return subprocess.run(['nice','-n','19',str(exe)],capture_output=True,text=True,timeout=15)
positive=run(basecode,'independent_bias_binding')
if positive.returncode:raise RuntimeError(positive.stderr)
mutations={
    'missing_static_calibration_guard':basecode.replace(' || !t.have_prediction_imu_calib','',1),
    'overwrite_sidecar_on_rejected_history_push':basecode.replace('if (t.slam_rels.push(rel, nts)) {','{ t.slam_rels.push(rel, nts);',1),
}
negative={}
for name,code in mutations.items():
    assert code!=basecode
    result=run(code,'negative_'+name)
    assert result.returncode==42,(name,result.returncode,result.stderr)
    negative[name]={'detected':True,'exit_code':result.returncode,'failure':result.stderr.strip()}
out={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'abi_header_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),'result':json.loads(positive.stdout),'negative_controls':negative,'scope':'Exact production read_pose_gyro_bias, visual_observation_count, flush_poses_locked, reset callback and gyro_bias_sample extracted. VIT transport, leaf math/writers and monotonic History push are deterministic stubs. Tests ABI guards, accepted timestamp association, rejected/duplicate/old/debug-skipped state, invalid/reacquired and successful/failed reset. Does not load backend or contact runtime.'}
assert source==src.read_text(),'Source changed during tests; rerun against frozen source'
(root/'independent-bias-binding.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
