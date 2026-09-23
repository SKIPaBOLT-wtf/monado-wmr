#!/usr/bin/env python3
# SPDX-License-Identifier: BSL-1.0
"""Offline extraction of production HID routing, with deterministic HID/clock/callback stubs."""
from pathlib import Path
import hashlib, json, subprocess, tempfile, atexit

REPO = Path(__file__).resolve().parents[2]
_work = tempfile.TemporaryDirectory(prefix='g2-hid-test-')
atexit.register(_work.cleanup)
ROOT = Path(_work.name)
SOURCE = REPO / 'src/xrt/drivers/wmr'
def extract(path, marker):
    s = path.read_text(); start = s.index(marker); brace = s.index('{', start); depth = 0
    for i in range(brace, len(s)):
        depth += (s[i] == '{') - (s[i] == '}')
        if not depth: return s[start:i+1]
    raise ValueError(marker)

paths = [SOURCE/n for n in ('wmr_hmd.c', 'wmr_hmd.h', 'wmr_hmd_controller.c', 'wmr_controller_base.c')]
hp, hh, cp, bp = paths
protocol = SOURCE/'wmr_controller_protocol.h'
prefix = r'''
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#define WMR_TRACE(...) ((void)0)
#define WMR_DEBUG(...) ((void)0)
#define WMR_WARN(...) ((void)0)
#define WMR_ERROR(...) ((void)0)
#define DRV_TRACE_MARKER() ((void)0)
#define WMR_PACKED __attribute__((packed))
#define __le32 uint32_t
#define __le32_to_cpu(x) (x)
#define U_TIME_1MS_IN_NS 1000000
#define WMR_FEATURE_BUFFER_SIZE 497
#define WMR_PENDING_REPORT_CAPACITY 128
#define WMR_MS_HOLOLENS_MSG_SENSORS 1
#define WMR_MS_HOLOLENS_MSG_CONTROL 2
#define WMR_MS_HOLOLENS_MSG_DEBUG 3
#define WMR_MS_HOLOLENS_MSG_BT_IFACE 5
#define WMR_MS_HOLOLENS_MSG_LEFT_CONTROLLER 6
#define WMR_MS_HOLOLENS_MSG_RIGHT_CONTROLLER 14
#define WMR_MS_HOLOLENS_MSG_CONTROLLER_STATUS 23
'''
types = extract(hh, 'struct wmr_pending_report\n') + ';\n' + r'''
struct wmr_controller_connection {int id;};
struct wmr_hmd_controller_connection;
struct wmr_hmd {int hid_lock; void *hid_hololens_sensors_dev;
 struct wmr_hmd_controller_connection *controller[2];
 struct wmr_pending_report pending_reports[WMR_PENDING_REPORT_CAPACITY];
 uint32_t pending_report_head,pending_report_count;
};
struct wmr_hmd_controller_connection {struct wmr_controller_connection base; int lock,cond; bool disconnected; int busy; uint8_t hmd_cmd_base; struct wmr_hmd *hmd;};
struct wmr_controller_base {int conn_lock; struct wmr_controller_connection *wcc;};
static void os_mutex_lock(int *m){assert(*m==0);*m=1;}
static void os_mutex_unlock(int *m){assert(*m==1);*m=0;}
static void os_cond_signal(int *c){(void)c;}
static int64_t clock_ns;
static int64_t os_monotonic_get_ns(void){return clock_ns;}
struct packet {int size; int64_t delay; uint8_t data[497];};
static struct packet queue[1024];
static int count,at,reads,truncated,sensors,inputs,statuses,other,callback_count,groups;
static bool send_ok=true,inject_status;
static struct wmr_hmd *active_hmd;
static struct wmr_controller_base *active_base;
static int callback_kind[2048]; static uint64_t callback_time[2048]; static uint8_t callback_tag[2048];
static void callback(int kind,uint64_t when,uint8_t tag){assert(active_hmd->hid_lock==0);callback_kind[callback_count]=kind;callback_time[callback_count]=when;callback_tag[callback_count++]=tag;}
static int os_hid_read(void *dev,uint8_t *out,size_t capacity,int timeout_ms){
 (void)dev; assert(active_hmd->hid_lock==1); assert(timeout_ms>=0); reads++;
 if(at>=count){clock_ns+=(int64_t)timeout_ms*1000000;return 0;}
 struct packet *p=&queue[at];
 if(p->delay>(int64_t)timeout_ms*1000000){p->delay-=(int64_t)timeout_ms*1000000;clock_ns+=(int64_t)timeout_ms*1000000;return 0;}
 at++;clock_ns+=p->delay;if(p->size<=0)return p->size;
 if(capacity<(size_t)p->size)truncated++;
 size_t n=capacity<(size_t)p->size?capacity:(size_t)p->size;memcpy(out,p->data,n);return (int)n;
}
static void hololens_handle_sensors(struct wmr_hmd *h,const uint8_t *b,int n){assert(h==active_hmd);assert(n==381||n==497);assert(b[n-1]==0xa5);sensors++;callback(1,clock_ns,b[1]);}
static void wmr_controller_connection_receive_bytes(struct wmr_controller_connection *c,uint64_t ts,uint8_t *b,int n){assert(c);assert(n>=45);inputs++;callback(b[0],ts,b[1]);}
static void misc(struct wmr_hmd *h,const uint8_t *b,int n){assert(h==active_hmd);assert(n>0);other++;callback(b[0],clock_ns,b[1]);}
#define hololens_handle_bt_iface_packet misc
#define hololens_handle_control misc
#define hololens_handle_debug misc
#define hololens_handle_unknown misc
static bool hololens_defer_packet(struct wmr_hmd*,const unsigned char*,int,uint64_t);
static void hololens_handle_controller_status_packet(struct wmr_hmd *h,const uint8_t *b,int n){
 assert(n>0);statuses++;assert(active_base->conn_lock==0);callback(23,clock_ns,b[1]);
 if(inject_status){inject_status=false;uint8_t saved=b[1];uint8_t fresh[3]={3,201,0};assert(hololens_defer_packet(h,fresh,3,clock_ns));assert(b[1]==saved);}
}
static bool wmr_controller_send_bytes(struct wmr_controller_base *w,const uint8_t *b,uint32_t n){(void)w;(void)b;(void)n;return send_ok;}
'''
funcs = [extract(hp, 'static void\nhololens_handle_controller_packet('),
         extract(hp, 'static void\nhololens_dispatch_packet('),
         extract(hp, 'static bool\nhololens_defer_packet('),
         extract(hp, 'static bool\nhololens_sensors_read_packets('),
         extract(hp, 'int\nwmr_hmd_read_sync_from_controller('),
         extract(cp, 'static int\nread_sync_from_controller(')]
bridge = r'''
static int wmr_controller_connection_read_sync(struct wmr_controller_connection *c,uint8_t *b,uint32_t n,int t){return read_sync_from_controller(c,b,n,t);}
'''
tail_funcs = [extract(bp, 'static int\nwmr_controller_read_sync('), extract(bp, 'static int\nwmr_controller_send_fw_cmd(')]
main = r'''
static struct wmr_hmd hmd;static struct wmr_hmd_controller_connection conn,other_conn;static struct wmr_controller_base base;
static struct wmr_controller_fw_cmd cmd;static struct wmr_controller_fw_cmd_response reply;
static void reset(uint8_t channel){memset(&hmd,0,sizeof(hmd));memset(&conn,0,sizeof(conn));memset(&other_conn,0,sizeof(other_conn));memset(&base,0,sizeof(base));memset(queue,0,sizeof(queue));memset(&reply,0xcc,sizeof(reply));memset(&cmd,0,sizeof(cmd));clock_ns=0;count=at=reads=truncated=sensors=inputs=statuses=other=callback_count=0;send_ok=true;inject_status=false;active_hmd=&hmd;active_base=&base;conn.hmd=&hmd;conn.hmd_cmd_base=channel;base.wcc=&conn.base;cmd.cmd.cmd_id=2;}
static void add(int size,uint8_t type,uint8_t tag,int64_t delay){struct packet *p=&queue[count++];p->size=size;p->delay=delay;if(size>0){memset(p->data,0,(size_t)size);p->data[0]=type;if(size>1)p->data[1]=tag;if(size>2)p->data[size-1]=0xa5;}}
static void fw(uint8_t channel,int code,int echo,int size,int64_t delay){add(size,(uint8_t)(channel+code),0,delay);queue[count-1].data[2]=(uint8_t)echo;queue[count-1].data[20]=0xab;}
static int runfw(void){return wmr_controller_send_fw_cmd(&base,&cmd,2,&reply);}
static void drained(void){while(hmd.pending_report_count)assert(hololens_sensors_read_packets(&hmd));assert(!hmd.hid_lock&&!base.conn_lock&&!conn.lock&&conn.busy==0);}
int main(void){
 for(int ch=5;ch<=13;ch+=8){reset((uint8_t)ch);for(int i=0;i<60;i++)add(i%2?381:497,1,(uint8_t)i,4000000);fw((uint8_t)ch,2,2,78,4000000);assert(runfw()==78);assert(reply.buf[0]==2&&reply.buf[20]==0xab&&reply.response.cmd_id_echo==2);assert(sensors==60&&truncated==0&&at==61&&clock_ns==244000000);assert(hmd.pending_report_count==0);groups++;}
 reset(5);for(int i=0;i<63;i++)add(497,1,(uint8_t)i,4000000);assert(runfw()==-ETIMEDOUT);assert(clock_ns==250000000&&sensors==62&&at==62&&truncated==0);groups++;
 reset(5);int malformed[]={1,78,380,382,496};for(unsigned i=0;i<sizeof(malformed)/sizeof(malformed[0]);i++)add(malformed[i],1,0,0);add(381,1,0,0);add(497,1,0,0);fw(5,2,2,78,0);assert(runfw()==78&&sensors==2);groups++;
 for(int ch=5;ch<=13;ch+=8){reset((uint8_t)ch);int own=ch==5?0:1,oth=1-own;hmd.controller[own]=&conn;hmd.controller[oth]=&other_conn;add(45,(uint8_t)(ch+1),11,3000000);add(45,(uint8_t)(ch==5?14:6),22,3000000);add(8,23,33,3000000);fw((uint8_t)ch,2,2,78,3000000);assert(runfw()==78&&inputs==1&&statuses==0&&hmd.pending_report_count==2);assert(callback_kind[0]==(ch==5?14:6)&&callback_time[0]==6000000);clock_ns=900000000;drained();assert(inputs==2&&statuses==1&&callback_tag[1]==11&&callback_time[1]==3000000&&callback_tag[2]==33);groups++;}
 reset(5);add(45,14,7,1000000);fw(5,2,2,78,1000000);assert(runfw()==78&&inputs==0&&hmd.pending_report_count==1);hmd.controller[1]=&other_conn;clock_ns=30000000;drained();assert(inputs==1&&callback_time[0]==1000000);groups++;
 reset(5);for(int i=0;i<129;i++)add(8,23,(uint8_t)i,0);assert(runfw()==-1&&at==128&&hmd.pending_report_count==128&&statuses==0);int saved_reads=reads;assert(runfw()==-1&&reads==saved_reads&&at==128);inject_status=true;drained();assert(statuses==128&&other==1&&callback_count==129);for(int i=0;i<128;i++)assert(callback_tag[i]==i);assert(callback_tag[128]==201);assert(at==128);assert(hololens_sensors_read_packets(&hmd)&&statuses==129&&at==129);groups++;
 reset(5);fw(13,2,2,78,0);fw(5,6,2,78,0);fw(5,2,2,78,0);assert(runfw()==78&&hmd.pending_report_count==1);drained();assert(other==1);groups++;
 reset(5);fw(5,2,99,78,0);assert(runfw()==-1);reset(5);fw(5,2,2,77,0);assert(runfw()==-1);reset(5);fw(5,2,2,79,0);assert(runfw()==-1);groups++;
 reset(13);fw(13,6,2,78,0);uint8_t direct[78];assert(wmr_hmd_read_sync_from_controller(&hmd,13,direct,sizeof(direct),0)==78&&direct[0]==19);groups++;
 reset(5);add(497,1,9,0);fw(5,2,2,78,0);assert(wmr_hmd_read_sync_from_controller(&hmd,5,direct,sizeof(direct),0)==0&&sensors==1&&at==1);assert(wmr_hmd_read_sync_from_controller(&hmd,5,direct,sizeof(direct),0)==78&&at==2);groups++;
 reset(5);add(-1,0,0,0);assert(runfw()==-1&&reads==1);reset(5);add(-EIO,0,0,0);assert(wmr_hmd_read_sync_from_controller(&hmd,5,direct,sizeof(direct),250)==-EIO);reset(5);assert(wmr_hmd_read_sync_from_controller(&hmd,5,direct,sizeof(direct),0)==0&&clock_ns==0&&reads==1);groups++;
 reset(5);conn.disconnected=true;assert(runfw()==-1&&reads==0);reset(5);send_ok=false;assert(runfw()==-1&&reads==0);groups++;
 reset(5);add(-1,0,0,0);assert(!hololens_sensors_read_packets(&hmd));reset(5);assert(hololens_sensors_read_packets(&hmd)&&callback_count==0);groups++;
 reset(5);for(int i=0;i<10;i++)add(i%2?381:497,1,(uint8_t)i,0);for(int i=0;i<10;i++)assert(hololens_sensors_read_packets(&hmd));assert(sensors==10&&truncated==0);groups++;
 printf("{\"groups\":%d,\"all_passed\":true,\"sensor_packets_preserved_per_successful_channel\":60,\"firmware_deadline_ms\":250,\"deferred_capacity_preserved\":128}\n",groups);
}
'''
code = prefix+types+'\n'+extract(protocol,'struct wmr_controller_fw_cmd\n')+';\n'+extract(protocol,'struct wmr_controller_fw_cmd_response\n')+';\n'+'\n'.join(funcs)+bridge+'\n'.join(tail_funcs)+main
cfile=ROOT/'independent_hid_preservation.c';binary=ROOT/'independent_hid_preservation';cfile.write_text(code)
subprocess.run(['nice','-n','19','cc','-std=c11','-O0','-g','-Wall','-Wextra',str(cfile),'-o',str(binary)],check=True)
run=subprocess.run(['nice','-n','19',str(binary)],check=True,capture_output=True,text=True,timeout=15)
result={'source_files':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'result':json.loads(run.stdout),'scope':'Exact production routing/read/FW functions and pending struct, with synthetic HID, deterministic time, assertive lock stubs, and leaf callback recording. No hardware/runtime calls. Covers both firmware channels, 381/497 preservation, malformed sizes, other/current/unpublished controller routing, receive timestamps, FIFO wrap/reentry/capacity, matching echo/channel/length, absolute timeout and zero/error reads.'}
(ROOT/'independent-hid-preservation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
