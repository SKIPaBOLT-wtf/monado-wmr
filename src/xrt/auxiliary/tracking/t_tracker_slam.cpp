// Copyright 2021-2024, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief SLAM tracking code.
 * @author Mateo de Mayo <mateo.demayo@collabora.com>
 * @ingroup aux_tracking
 */

#include "xrt/xrt_config_have.h"
#include "xrt/xrt_defines.h"
#include "xrt/xrt_tracking.h"
#include "xrt/xrt_frameserver.h"

#include "util/u_debug.h"
#include "util/u_frame_ts_guard.h"
#include "util/u_g2_telemetry.h"
#include "util/u_world_reanchor.h"
#include "util/u_logging.h"
#include "util/u_misc.h"
#include "util/u_sink.h"
#include "util/u_var.h"
#include "util/u_trace_marker.h"

#include "os/os_threading.h"

#include "math/m_api.h"
#include "math/m_filter_fifo.h"
#include "math/m_filter_one_euro.h"
#include "math/m_predict.h"
#include "math/m_relation_history.h"
#include "math/m_space.h"
#include "math/m_vec3.h"

#include "tracking/t_euroc_recorder.h"
#include "tracking/t_openvr_tracker.h"
#include "tracking/t_tracking.h"
#include "tracking/t_vit_loader.h"
#include "tracking/t_dead_reckoning.h"

#include "vit/vit_interface.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/version.hpp>

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <vector>


//! @todo Get preferred system from systems found at build time
#define PREFERRED_VIT_SYSTEM_LIBRARY "libbasalt.so"

#define SLAM_TRACE(...) U_LOG_IFL_T(t.log_level, __VA_ARGS__)
#define SLAM_DEBUG(...) U_LOG_IFL_D(t.log_level, __VA_ARGS__)
#define SLAM_INFO(...) U_LOG_IFL_I(t.log_level, __VA_ARGS__)
#define SLAM_WARN(...) U_LOG_IFL_W(t.log_level, __VA_ARGS__)
#define SLAM_ERROR(...) U_LOG_IFL_E(t.log_level, __VA_ARGS__)
#define SLAM_ASSERT(predicate, ...)                                                                                    \
	do {                                                                                                           \
		bool p = predicate;                                                                                    \
		if (!p) {                                                                                              \
			U_LOG(U_LOGGING_ERROR, __VA_ARGS__);                                                           \
			assert(false && "SLAM_ASSERT failed: " #predicate);                                            \
			exit(EXIT_FAILURE);                                                                            \
		}                                                                                                      \
	} while (false);
#define SLAM_ASSERT_(predicate) SLAM_ASSERT(predicate, "Assertion failed " #predicate)

// Debug assertions, not vital but useful for finding errors
#ifdef NDEBUG
#define SLAM_DASSERT(predicate, ...) (void)(predicate)
#define SLAM_DASSERT_(predicate) (void)(predicate)
#else
#define SLAM_DASSERT(predicate, ...) SLAM_ASSERT(predicate, __VA_ARGS__)
#define SLAM_DASSERT_(predicate) SLAM_ASSERT_(predicate)
#endif

//! @see t_slam_tracker_config
DEBUG_GET_ONCE_LOG_OPTION(slam_log, "SLAM_LOG", U_LOGGING_INFO)
DEBUG_GET_ONCE_OPTION(vit_system_library_path, "VIT_SYSTEM_LIBRARY_PATH", PREFERRED_VIT_SYSTEM_LIBRARY)
DEBUG_GET_ONCE_OPTION(slam_config, "SLAM_CONFIG", nullptr)
DEBUG_GET_ONCE_BOOL_OPTION(slam_ui, "SLAM_UI", false)
DEBUG_GET_ONCE_BOOL_OPTION(slam_submit_from_start, "SLAM_SUBMIT_FROM_START", false)
DEBUG_GET_ONCE_NUM_OPTION(slam_openvr_groundtruth_device, "SLAM_OPENVR_GROUNDTRUTH_DEVICE", 0)
DEBUG_GET_ONCE_NUM_OPTION(slam_prediction_type, "SLAM_PREDICTION_TYPE", long(SLAM_PRED_DEAD_RECKONING))
DEBUG_GET_ONCE_BOOL_OPTION(slam_write_csvs, "SLAM_WRITE_CSVS", false)
DEBUG_GET_ONCE_OPTION(slam_csv_path, "SLAM_CSV_PATH", "evaluation/")
DEBUG_GET_ONCE_BOOL_OPTION(slam_timing_stat, "SLAM_TIMING_STAT", true)
DEBUG_GET_ONCE_BOOL_OPTION(slam_features_stat, "SLAM_FEATURES_STAT", true)
DEBUG_GET_ONCE_BOOL_OPTION(g2_require_visual_observations, "G2_REQUIRE_VISUAL_OBSERVATIONS", false)
DEBUG_GET_ONCE_BOOL_OPTION(g2_predict_with_vit_bias, "G2_PREDICT_WITH_VIT_BIAS", false)
DEBUG_GET_ONCE_NUM_OPTION(slam_cam_count, "SLAM_CAM_COUNT", 2)

//! Namespace for the interface to the external SLAM tracking system
namespace xrt::auxiliary::tracking::slam {
constexpr int UI_TIMING_POSE_COUNT = 192;
constexpr int UI_FEATURES_POSE_COUNT = 192;
constexpr int UI_GTDIFF_POSE_COUNT = 192;

using os::Mutex;
using std::deque;
using std::ifstream;
using std::make_shared;
using std::map;
using std::ofstream;
using std::ostream;
using std::pair;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::unique_lock;
using std::vector;
using std::filesystem::create_directories;
using Trajectory = map<timepoint_ns, xrt_pose>;
using timing_sample = vector<timepoint_ns>;

using xrt::auxiliary::math::RelationHistory;


/*
 *
 * CSV Writers
 *
 */

ostream &
operator<<(ostream &os, const xrt_pose_sample &s)
{
	timepoint_ns ts = s.timestamp_ns;
	xrt_vec3 p = s.pose.position;
	xrt_quat r = s.pose.orientation;
	os << ts << ",";
	os << p.x << "," << p.y << "," << p.z << ",";
	os << r.w << "," << r.x << "," << r.y << "," << r.z << CSV_EOL;
	return os;
}

ostream &
operator<<(ostream &os, const timing_sample &timestamps)
{
	for (const timepoint_ns &ts : timestamps) {
		string delimiter = &ts != &timestamps.back() ? "," : CSV_EOL;
		os << ts << delimiter;
	}
	return os;
}

struct feature_count_sample
{
	timepoint_ns ts;
	vector<int> counts;
};

struct gyro_bias_sample
{
	timepoint_ns ts;
	bool valid;
	xrt_vec3 bias;
};

ostream &
operator<<(ostream &os, const gyro_bias_sample &s)
{
	os << s.ts << "," << s.valid << "," << s.bias.x << "," << s.bias.y << "," << s.bias.z << CSV_EOL;
	return os;
}

ostream &
operator<<(ostream &os, const feature_count_sample &s)
{
	os << s.ts;
	for (int count : s.counts) {
		os << "," << count;
	}
	os << CSV_EOL;
	return os;
}

//! Writes a CSV file for a particular row type
template <typename RowType> class CSVWriter
{
public:
	bool enabled; // Modified through UI

protected:
	vector<string> column_names;

private:
	string directory;
	string filename;
	ofstream file;
	bool created = false;
	Mutex mutex;

	void
	create()
	{
		create_directories(directory);
		file = ofstream{directory + "/" + filename};
		file << "#";
		for (const string &col : column_names) {
			string delimiter = &col != &column_names.back() ? "," : CSV_EOL;
			file << col << delimiter;
		}
		file << std::fixed << std::setprecision(CSV_PRECISION);
	}

public:
	CSVWriter(const string &dir, const string &fn, bool e, const vector<string> &cn = {})
	    : enabled(e), column_names(cn), directory(dir), filename(fn)
	{}

	virtual ~CSVWriter() {}

	void
	push(const RowType &row)
	{
		unique_lock lock(mutex);

		if (!enabled) {
			return;
		}

		if (!created) {
			created = true;
			create();
		}

		file << row;
	}
};

//! Writes poses and their timestamps to a CSV file
struct TrajectoryWriter : public CSVWriter<xrt_pose_sample>
{
	TrajectoryWriter(const string &dir, const string &fn, bool e) : CSVWriter<xrt_pose_sample>(dir, fn, e)
	{
		column_names = {"timestamp [ns]", "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
		                "q_RS_w []",      "q_RS_x []",    "q_RS_y []",    "q_RS_z []"};
	}
};

//! Writes timestamps measured when estimating a new pose by the SLAM system
struct TimingWriter : public CSVWriter<timing_sample>
{
	TimingWriter(const string &dir, const string &fn, bool e, const vector<string> &cn)
	    : CSVWriter<timing_sample>(dir, fn, e, cn)
	{}
};

//! Writes feature information specific to a particular estimated pose
struct FeaturesWriter : public CSVWriter<feature_count_sample>
{
	FeaturesWriter(const string &dir, const string &fn, bool e, size_t cam_count)
	    : CSVWriter<feature_count_sample>(dir, fn, e)
	{
		column_names.push_back("timestamp");
		for (size_t i = 0; i < cam_count; i++) {
			column_names.push_back("cam" + to_string(i) + " feature count");
		}
	}
};

/*!
 * Main implementation of @ref xrt_tracked_slam. This is an adapter class for
 * SLAM tracking that wraps an external SLAM implementation.
 *
 * @implements xrt_tracked_slam
 * @implements xrt_frame_node
 * @implements xrt_frame_sink
 * @implements xrt_imu_sink
 * @implements xrt_pose_sink
 */
struct TrackerSlam
{
	struct xrt_tracked_slam base = {};
	struct xrt_frame_node node = {};       //!< Will be called on destruction
	struct t_vit_bundle vit;               //!< VIT system function pointers
	struct vit_tracker_extension_set exts; //!< VIT tracker supported extensions
	struct vit_tracker *tracker;           //!< Pointer to the tracker created by the loaded VIT system;

	struct xrt_slam_sinks sinks = {};                            //!< Pointers to the sinks below
	struct xrt_frame_sink cam_sinks[XRT_TRACKING_MAX_SLAM_CAMS]; //!< Sends camera frames to the SLAM system
	struct xrt_imu_sink imu_sink = {};                           //!< Sends imu samples to the SLAM system
	struct xrt_pose_sink gt_sink = {};                           //!< Register groundtruth trajectory for stats
	struct xrt_device_masks_sink hand_masks_sink = {};           //!< Register latest hand masks to ignore
	struct xrt_device_masks_sink controller_masks_sink = {};     //!< Register latest controller masks to ignore

	bool submit;        //!< Whether to submit data pushed to sinks to the SLAM tracker
	uint32_t cam_count; //!< Number of cameras used for tracking

	struct u_var_button reset_state_btn; //!< Reset tracker state button

	enum u_logging_level log_level; //!< Logging level for the SLAM tracker, set by SLAM_LOG var

	struct xrt_slam_sinks *euroc_recorder; //!< EuRoC dataset recording sinks
	struct openvr_tracker *ovr_tracker;    //!< OpenVR lighthouse tracker

	// Used mainly for checking that the timestamps come in order
	timepoint_ns last_imu_ts; //!< Last received IMU sample timestamp

	//! Per-camera timestamp guards: frames that would hand the SLAM system a non-monotonic or
	//! implausible timeline are dropped here (Basalt aborts the process on any regression).
	struct u_frame_ts_guard cam_ts_guard[XRT_TRACKING_MAX_SLAM_CAMS];
	timepoint_ns last_ts_drop_warn_mono; //!< Rate limiter for the guard-drop warning
	uint64_t ts_drop_count;              //!< Total frames dropped by the guards

	struct xrt_device_masks_sample last_hand_masks;       //!< Last received hand masks info
	Mutex last_hand_masks_mutex;                          //!< Mutex for @ref last_hand_masks
	struct xrt_device_masks_sample last_controller_masks; //!< Last received hand masks info
	Mutex last_controller_masks_mutex;                    //!< Mutex for @ref last_controller_masks

	// Prediction

	//! Type of prediction to use
	t_slam_prediction_type pred_type;
	u_var_combo pred_combo;         //!< UI combo box to select @ref pred_type
	Mutex pose_mutex; //!< serializes VIT queue draining, prediction/filter output and explicit reset
	int slam_consec_bad = 0; //!< guarded by pose_mutex, owned by this tracker
	bool require_visual_observations = false; //!< explicit G2 opt-in; zero current-frame observations invalidate pose
	bool visual_observations_available = false; //!< extension successfully enabled before streaming
	bool use_pose_gyro_bias = false; //!< Optional vendor ABI; calibrated IMU-frame gyro only.
	gyro_bias_sample latest_gyro_bias{}; //!< Belongs to the actually accepted latest relation, under pose_mutex.
	RelationHistory slam_rels{};    //!< A history of relations produced purely from external SLAM tracker data
	int dbg_pred_every = 1;         //!< Skip X SLAM poses so that you get tracked mostly by the prediction algo
	int dbg_pred_counter = 0;       //!< SLAM pose counter for prediction debugging
	struct os_mutex lock_ff;        //!< Lock for gyro_ff and accel_ff.
	struct m_ff_vec3_f32 *gyro_ff;  //!< Last gyroscope samples
	struct m_ff_vec3_f32 *accel_ff; //!< Last accelerometer samples
	struct t_imu_calibration prediction_imu_calib{}; //!< Driver calibration accepted by the tracker.
	bool have_prediction_imu_calib = false;
	vector<u_sink_debug> ui_sink;   //!< Sink to display frames in UI of each camera

	//! Used to correct accelerometer measurements when integrating into the prediction.
	//! @todo Should be automatically computed instead of required to be filled manually through the UI.
	xrt_vec3 gravity_correction{0, 0, -MATH_GRAVITY_M_S2};


	//! Filters are used to smooth out the resulting trajectory
	struct
	{
		// Moving average filter
		bool use_moving_average_filter = false;
		//! Time window in ms take the average on.
		//! Increasing it smooths out the tracking at the cost of adding delay.
		double window = 66;
		struct m_ff_vec3_f32 *pos_ff; //! Predicted positions fifo
		struct m_ff_vec3_f32 *rot_ff; //! Predicted rotations fifo (only xyz components, w is inferred)

		// Exponential smoothing filter
		bool use_exponential_smoothing_filter = false;
		float alpha = 0.1; //!< How much should we lerp towards the @p target value on each update
		struct xrt_space_relation last = XRT_SPACE_RELATION_ZERO;   //!< Last filtered relation
		struct xrt_space_relation target = XRT_SPACE_RELATION_ZERO; //!< Target relation

		// One euro filter
		bool use_one_euro_filter = false;
		m_filter_euro_vec3 pos_oe;     //!< One euro position filter
		m_filter_euro_quat rot_oe;     //!< One euro rotation filter
		const float min_cutoff = M_PI; //!< Default minimum cutoff frequency
		const float min_dcutoff = 1;   //!< Default minimum cutoff frequency for the derivative
		const float beta = 0.16;       //!< Default speed coefficient

	} filter;

	// Stats and metrics

	// CSV writers for offline analysis (using pointers because of container_of)
	TimingWriter *slam_times_writer;      //!< Timestamps of the pipeline for performance analysis
	FeaturesWriter *slam_features_writer; //!< Feature tracking information for analysis
	CSVWriter<gyro_bias_sample> *gyro_bias_writer; //!< Optional diagnostics; no fitted or synthetic bias.
	TrajectoryWriter *slam_traj_writer;   //!< Estimated poses from the SLAM system
	TrajectoryWriter *pred_traj_writer;   //!< Predicted poses
	TrajectoryWriter *filt_traj_writer;   //!< Predicted and filtered poses

	//! Tracker timing info for performance evaluation
	struct
	{
		bool enabled = false;               //!< Whether the timing extension is enabled
		float dur_ms[UI_TIMING_POSE_COUNT]; //!< Timing durations in ms
		int idx = 0;                        //!< Index of latest entry in @p dur_ms
		u_var_combo start_ts;               //!< UI combo box to select initial timing measurement
		u_var_combo end_ts;                 //!< UI combo box to select final timing measurement
		int start_ts_idx;                   //!< Selected initial timing measurement in @p start_ts
		int end_ts_idx;                     //!< Selected final timing measurement in @p end_ts
		struct u_var_timing ui;             //!< Realtime UI for tracker durations
		vector<string> columns;             //!< Column names of the measured timestamps
		string joined_columns;              //!< Column names as a null separated string
		struct u_var_button enable_btn;     //!< Toggle tracker timing reports
	} timing;

	//! Tracker feature tracking info
	struct Features
	{
		struct FeatureCounter
		{
			//! Feature count for each frame timestamp for this camera.
			//! @note Harmless race condition over this as the UI might read this while it's being written
			deque<pair<timepoint_ns, int>> entries{};

			//! Persistently stored camera name for display in the UI
			string cam_name;

			void
			addFeatureCount(timepoint_ns ts, int count)
			{
				entries.emplace_back(ts, count);
				if (entries.size() > UI_FEATURES_POSE_COUNT) {
					entries.pop_front();
				}
			}
		};

		vector<FeatureCounter> fcs; //!< Store feature count info for each camera
		u_var_curves fcs_ui;        //!< Display of `fcs` in UI

		bool enabled = false;           //!< Whether the features extension is enabled
		struct u_var_button enable_btn; //!< Toggle extension
	} features;

	//! Ground truth related fields
	struct
	{
		Trajectory *trajectory;               //!< Empty if we've not received groundtruth
		xrt_pose origin;                      //!< First ground truth pose
		float diffs_mm[UI_GTDIFF_POSE_COUNT]; //!< Positional error wrt ground truth
		int diff_idx = 0;                     //!< Index of last error in @p diffs_mm
		struct u_var_timing diff_ui;          //!< Realtime UI for positional error
		bool override_tracking = false;       //!< Force the tracker to report gt poses instead
	} gt;
};


/*
 *
 * Timing functionality
 *
 */

static void
timing_ui_setup(TrackerSlam &t)
{
	t.timing.enabled = false;

	u_var_add_ro_raw_text(&t, "\nTracker timing", "Tracker timing");

	// Setup toggle button
	static const char *msg[2] = {"[OFF] Enable timing", "[ON] Disable timing"};
	u_var_button_cb cb = [](void *t_ptr) {
		TrackerSlam *t = (TrackerSlam *)t_ptr;
		u_var_button &btn = t->timing.enable_btn;
		bool e = !t->timing.enabled;
		snprintf(btn.label, sizeof(btn.label), "%s", msg[e]);
		vit_result_t vres = t->vit.tracker_enable_extension(t->tracker, VIT_TRACKER_EXTENSION_POSE_TIMING, e);
		if (vres != VIT_SUCCESS) {
			U_LOG_IFL_E(t->log_level, "Failed to set tracker timing extension");
			return;
		}
		t->timing.enabled = e;
	};
	t.timing.enable_btn.cb = cb;
	t.timing.enable_btn.disabled = !t.exts.has_pose_timing;
	t.timing.enable_btn.ptr = &t;
	u_var_add_button(&t, &t.timing.enable_btn, msg[t.timing.enabled]);

	// We provide two timing columns by default, even if there is no extension support
	t.timing.columns = {"sampled", "received_by_monado"};

	// Only fill the timing columns if the tracker supports pose timing
	if (t.exts.has_pose_timing) {
		vit_tracker_timing_titles titles = {};
		vit_result_t vres = t.vit.tracker_get_timing_titles(t.tracker, &titles);
		if (vres != VIT_SUCCESS) {
			SLAM_ERROR("Failed to get timing titles from tracker");
			return;
		}

		// Copies the titles locally.
		std::vector<std::string> cols(titles.titles, titles.titles + titles.count);

		t.timing.columns.insert(t.timing.columns.begin() + 1, cols.begin(), cols.end());
	}

	// Construct null-separated array of options for the combo box
	using namespace std::string_literals;
	t.timing.joined_columns = "";
	for (const string &name : t.timing.columns) {
		t.timing.joined_columns += name + "\0"s;
	}
	t.timing.joined_columns += "\0"s;

	t.timing.start_ts.count = t.timing.columns.size();
	t.timing.start_ts.options = t.timing.joined_columns.c_str();
	t.timing.start_ts.value = &t.timing.start_ts_idx;
	t.timing.start_ts_idx = 0;
	u_var_add_combo(&t, &t.timing.start_ts, "Start timestamp");

	t.timing.end_ts.count = t.timing.columns.size();
	t.timing.end_ts.options = t.timing.joined_columns.c_str();
	t.timing.end_ts.value = &t.timing.end_ts_idx;
	t.timing.end_ts_idx = t.timing.columns.size() - 1;
	u_var_add_combo(&t, &t.timing.end_ts, "End timestamp");

	t.timing.ui.values.data = t.timing.dur_ms;
	t.timing.ui.values.length = UI_TIMING_POSE_COUNT;
	t.timing.ui.values.index_ptr = &t.timing.idx;
	t.timing.ui.reference_timing = 16.6;
	t.timing.ui.center_reference_timing = true;
	t.timing.ui.range = t.timing.ui.reference_timing;
	t.timing.ui.dynamic_rescale = true;
	t.timing.ui.unit = "ms";
	u_var_add_f32_timing(&t, &t.timing.ui, "External tracker times");
}

//! Updates timing UI with info from a computed pose and returns that info
static vector<timepoint_ns>
timing_ui_push(TrackerSlam &t, const vit_pose_t *pose, int64_t ts)
{
	timepoint_ns now = os_monotonic_get_ns();
	vector<timepoint_ns> tss = {ts, now};

	// Add extra timestamps if the SLAM tracker provides them
	if (t.timing.enabled) {
		vit_pose_timing timing;
		vit_result_t vres = t.vit.pose_get_timing(pose, &timing);
		if (vres != VIT_SUCCESS) {
			// Even if the timing is enabled, some of the poses already in the queue won't have it enabled.
			if (vres != VIT_ERROR_NOT_ENABLED) {
				SLAM_ERROR("Failed to get pose timing");
			}

			return {};
		}

		std::vector<int64_t> data(timing.timestamps, timing.timestamps + timing.count);
		tss.insert(tss.begin() + 1, data.begin(), data.end());

		// The two timestamps to compare in the graph
		timepoint_ns start = tss.at(t.timing.start_ts_idx);
		timepoint_ns end = tss.at(t.timing.end_ts_idx);

		// Push to the UI graph
		float tss_ms = (end - start) / U_TIME_1MS_IN_NS;
		t.timing.idx = (t.timing.idx + 1) % UI_TIMING_POSE_COUNT;
		t.timing.dur_ms[t.timing.idx] = tss_ms;
		constexpr float a = 1.0f / UI_TIMING_POSE_COUNT; // Exponential moving average
		t.timing.ui.reference_timing = (1 - a) * t.timing.ui.reference_timing + a * tss_ms;
	}

	return tss;
}


/*
 *
 * Feature information functionality
 *
 */

static void
features_ui_setup(TrackerSlam &t)
{
	t.features.enabled = false;

	u_var_add_ro_raw_text(&t, "\nTracker features", "Tracker features");

	// Setup toggle button
	static const char *msg[2] = {"[OFF] Enable features info", "[ON] Disable features info"};
	u_var_button_cb cb = [](void *t_ptr) {
		TrackerSlam *t = (TrackerSlam *)t_ptr;
		u_var_button &btn = t->features.enable_btn;
		bool e = !t->features.enabled;
		snprintf(btn.label, sizeof(btn.label), "%s", msg[e]);
		vit_result_t vres = t->vit.tracker_enable_extension(t->tracker, VIT_TRACKER_EXTENSION_POSE_FEATURES, e);
		if (vres != VIT_SUCCESS) {
			U_LOG_IFL_E(t->log_level, "Failed to set tracker features extension");
			return;
		}
		t->features.enabled = e;
	};
	t.features.enable_btn.cb = cb;
	t.features.enable_btn.disabled = !t.exts.has_pose_features;
	t.features.enable_btn.ptr = &t;
	u_var_add_button(&t, &t.features.enable_btn, msg[t.features.enabled]);

	// Setup graph
	u_var_curve_getter getter = [](void *fs_ptr, int i) -> u_var_curve_point {
		auto *fs = (TrackerSlam::Features::FeatureCounter *)fs_ptr;
		timepoint_ns now = os_monotonic_get_ns();

		size_t size = fs->entries.size();
		if (size == 0) {
			return {0, 0};
		}

		int last_idx = size - 1;
		if (i > last_idx) {
			i = last_idx;
		}

		auto [ts, count] = fs->entries.at(last_idx - i);
		return {time_ns_to_s(now - ts), double(count)};
	};

	t.features.fcs_ui.curve_count = t.cam_count;
	t.features.fcs_ui.xlabel = "Last seconds";
	t.features.fcs_ui.ylabel = "Number of features";

	t.features.fcs.resize(t.cam_count);
	for (uint32_t i = 0; i < t.cam_count; ++i) {
		auto &fc = t.features.fcs[i];
		fc.cam_name = "Cam" + to_string(i);

		auto &fc_ui = t.features.fcs_ui.curves[i];
		fc_ui.count = UI_FEATURES_POSE_COUNT;
		fc_ui.data = &fc;
		fc_ui.getter = getter;
		fc_ui.label = fc.cam_name.c_str();
	}

	u_var_add_curves(&t, &t.features.fcs_ui, "Feature count");
}

static vector<int>
features_ui_push(TrackerSlam &t, const vit_pose_t *pose, int64_t ts)
{
	if (!t.features.enabled) {
		return {};
	}

	// Push to the UI graph
	vector<int> fcs{};
	for (uint32_t i = 0; i < t.cam_count; ++i) {
		vit_pose_features features = {};
		vit_result_t vres = t.vit.pose_get_features(pose, i, &features);
		if (vres != VIT_SUCCESS) {
			// Even if the features are enabled, some of the poses already in the queue won't have it
			// enabled.
			if (vres != VIT_ERROR_NOT_ENABLED) {
				SLAM_ERROR("Failed to get pose features for camera %u", i);
			}

			return {};
		}

		t.features.fcs.at(i).addFeatureCount(ts, features.count);
		fcs.push_back(features.count);
	}

	return fcs;
}

/*
 *
 * Ground truth functionality
 *
 */

//! Gets an interpolated groundtruth pose (if available) at a specified timestamp
static xrt_pose
get_gt_pose_at(const Trajectory &gt, timepoint_ns ts)
{
	if (gt.empty()) {
		return XRT_POSE_IDENTITY;
	}

	Trajectory::const_iterator rit = gt.upper_bound(ts);

	if (rit == gt.begin()) { // Too far in the past, return first gt pose
		return gt.begin()->second;
	}

	if (rit == gt.end()) { // Too far in the future, return last gt pose
		return std::prev(gt.end())->second;
	}

	Trajectory::const_iterator lit = std::prev(rit);

	const auto &[lts, lpose] = *lit;
	const auto &[rts, rpose] = *rit;

	float t = double(ts - lts) / double(rts - lts);
	SLAM_DASSERT_(0 <= t && t <= 1);

	xrt_pose res{};
	math_quat_slerp(&lpose.orientation, &rpose.orientation, t, &res.orientation);
	res.position = m_vec3_lerp(lpose.position, rpose.position, t);
	return res;
}

//! Converts a pose from the tracker to ground truth
static struct xrt_pose
xr2gt_pose(const xrt_pose &gt_origin, const xrt_pose &xr_pose)
{
	//! @todo Right now this is hardcoded for Basalt and the EuRoC vicon datasets
	//! groundtruth and ignores orientation. Applies a fixed transformation so
	//! that the tracked and groundtruth trajectories origins and general motion
	//! match. The usual way of evaluating trajectory errors in SLAM requires to
	//! first align the trajectories through a non-linear optimization (e.g. gauss
	//! newton) so that they are as similar as possible. For this you need the
	//! entire tracked trajectory to be known beforehand, which makes it not
	//! suitable for reporting an error metric in realtime. See this 2-page paper
	//! for more info on trajectory alignment:
	//! https://ylatif.github.io/movingsensors/cameraReady/paper07.pdf

	xrt_vec3 pos = xr_pose.position;
	xrt_quat z180{0, 0, 1, 0};
	math_quat_rotate_vec3(&z180, &pos, &pos);
	math_quat_rotate_vec3(&gt_origin.orientation, &pos, &pos);
	pos += gt_origin.position;

	return {XRT_QUAT_IDENTITY, pos};
}

//! The inverse of @ref xr2gt_pose.
static struct xrt_pose
gt2xr_pose(const xrt_pose &gt_origin, const xrt_pose &gt_pose)
{
	xrt_vec3 pos = gt_pose.position;
	pos -= gt_origin.position;
	xrt_quat gt_origin_orientation_inv = gt_origin.orientation;
	math_quat_invert(&gt_origin_orientation_inv, &gt_origin_orientation_inv);
	math_quat_rotate_vec3(&gt_origin_orientation_inv, &pos, &pos);
	xrt_quat zn180{0, 0, -1, 0};
	math_quat_rotate_vec3(&zn180, &pos, &pos);

	return {XRT_QUAT_IDENTITY, pos};
}

static void
gt_ui_setup(TrackerSlam &t)
{
	u_var_add_ro_raw_text(&t, "\nTracker groundtruth", "Tracker groundtruth");
	t.gt.diff_ui.values.data = t.gt.diffs_mm;
	t.gt.diff_ui.values.length = UI_GTDIFF_POSE_COUNT;
	t.gt.diff_ui.values.index_ptr = &t.gt.diff_idx;
	t.gt.diff_ui.reference_timing = 0;
	t.gt.diff_ui.center_reference_timing = true;
	t.gt.diff_ui.range = 100; // 10cm
	t.gt.diff_ui.dynamic_rescale = true;
	t.gt.diff_ui.unit = "mm";
	u_var_add_f32_timing(&t, &t.gt.diff_ui, "Tracking absolute error");
}

static void
gt_ui_push(TrackerSlam &t, timepoint_ns ts, xrt_pose tracked_pose)
{
	if (t.gt.trajectory->empty()) {
		return;
	}

	xrt_pose gt_pose = get_gt_pose_at(*t.gt.trajectory, ts);
	xrt_pose xr_pose = xr2gt_pose(t.gt.origin, tracked_pose);

	float len_mm = m_vec3_len(xr_pose.position - gt_pose.position) * 1000;
	t.gt.diff_idx = (t.gt.diff_idx + 1) % UI_GTDIFF_POSE_COUNT;
	t.gt.diffs_mm[t.gt.diff_idx] = len_mm;
	constexpr float a = 1.0f / UI_GTDIFF_POSE_COUNT; // Exponential moving average
	t.gt.diff_ui.reference_timing = (1 - a) * t.gt.diff_ui.reference_timing + a * len_mm;
}

/*
 *
 * Tracker functionality
 *
 */

//! -1 means unavailable/unknown, never zero. Positive counts are not a quality guarantee.
//! Basalt's feature extension enumerates landmark observations for this pose's current frame.
static int64_t
visual_observation_count(TrackerSlam &t, const vit_pose_t *pose)
{
	if (!t.require_visual_observations || !t.visual_observations_available || t.cam_count == 0) {
		return -1;
	}
	int64_t count = 0;
	for (uint32_t i = 0; i < t.cam_count; ++i) {
		vit_pose_features features{};
		if (t.vit.pose_get_features(pose, i, &features) != VIT_SUCCESS) {
			return -1;
		}
		count += features.count;
	}
	return count;
}

//! Read a copied bias from the same immutable backend pose used by pose_get_data.
static gyro_bias_sample
read_pose_gyro_bias(TrackerSlam &t, const vit_pose_t *pose, int64_t ts)
{
	gyro_bias_sample sample{ts, false, {}};
	if (!t.use_pose_gyro_bias || !t.have_prediction_imu_calib || t.vit.g2_pose_get_imu_bias_v1 == nullptr) {
		return sample;
	}
	vit_g2_imu_bias_v1 data{};
	if (t.vit.g2_pose_get_imu_bias_v1(pose, sizeof(data), &data) != VIT_SUCCESS ||
	    data.struct_size != sizeof(data) || data.timestamp_ns != ts ||
	    (data.flags & 1u) == 0 || (data.flags & ~3u) != 0) {
		return sample;
	}
	for (double v : data.gyro_bias_rad_s) {
		if (!std::isfinite(v) || std::abs(v) > FLT_MAX) {
			return sample;
		}
	}
	sample.bias = {(float)data.gyro_bias_rad_s[0], (float)data.gyro_bias_rad_s[1],
	               (float)data.gyro_bias_rad_s[2]};
	sample.valid = true;
	return sample;
}

//! Dequeue all tracked poses from the SLAM system and update prediction data with them.
static bool
flush_poses_locked(TrackerSlam &t)
{
	// Caller holds pose_mutex across pop, relation publication and dependent output state.

	vit_pose_t *pose = NULL;
	vit_result_t vres = t.vit.tracker_pop_pose(t.tracker, &pose);
	if (vres != VIT_SUCCESS) {
		SLAM_ERROR("Failed to get pose from VIT tracker");
	}

	if (pose == NULL) {
		SLAM_TRACE("No poses to flush");
		return false;
	}

	do {
		// New pose
		vit_pose_data_t data;
		vres = t.vit.pose_get_data(pose, &data);
		if (vres != VIT_SUCCESS) {
			SLAM_ERROR("Failed to get pose data from VIT tracker");
			return false;
		}

		int64_t nts = data.timestamp;
		const bool no_visual_observations = visual_observation_count(t, pose) == 0;

		xrt_vec3 npos{data.px, data.py, data.pz};
		xrt_quat nrot{data.ox, data.oy, data.oz, data.ow};
		xrt_vec3 nvel{data.vx, data.vy, data.vz};

		// Divergence guard: Basalt can emit a non-finite or wildly-displaced pose when visual
		// landmarks degenerate (a saturated window -> backsubstitution NaN). Pushing it permanently
		// corrupts the relation history (the head then "diverges forever until reset"). Drop bad
		// poses; after a sustained run, reset the tracker so the head self-heals. Head-side analogue
		// of the controller optical-jump gate.
		static const int SLAM_MAX_CONSEC_BAD = 30; // ~1 s of bad poses before forcing a reset
		bool pose_finite = std::isfinite(npos.x) && std::isfinite(npos.y) && std::isfinite(npos.z) &&
		                   std::isfinite(nrot.x) && std::isfinite(nrot.y) && std::isfinite(nrot.z) &&
		                   std::isfinite(nrot.w) && std::isfinite(nvel.x) && std::isfinite(nvel.y) &&
		                   std::isfinite(nvel.z);
		bool pose_insane = pose_finite && m_vec3_len(npos) > 1000.0f; // room-scale; >1 km = diverged
		if (!pose_finite || pose_insane) {
			if (no_visual_observations) {
				xrt_space_relation unavailable{};
				unavailable.pose.orientation.w = 1;
				if (t.slam_rels.push(unavailable, nts)) {
					t.latest_gyro_bias = {nts, false, {}};
					t.gyro_bias_writer->push(t.latest_gyro_bias);
				}
			}
			t.slam_consec_bad++;
			SLAM_WARN("Dropping divergent SLAM pose (finite=%d |p|=%.1f m) %d/%d", pose_finite,
			          pose_finite ? (double)m_vec3_len(npos) : 0.0, t.slam_consec_bad, SLAM_MAX_CONSEC_BAD);
			if (t.slam_consec_bad >= SLAM_MAX_CONSEC_BAD) {
				t.latest_gyro_bias = {};
				if (t.vit.tracker_reset(t.tracker) != VIT_SUCCESS) {
					SLAM_WARN("Auto-reset of diverged VIT tracker failed");
				} else {
					SLAM_INFO("Auto-reset VIT tracker after sustained divergence");
				}
				t.slam_consec_bad = 0;
			}
			t.vit.pose_destroy(pose);
			continue;
		}
		t.slam_consec_bad = 0;

		// Last relation
		xrt_space_relation lr = XRT_SPACE_RELATION_ZERO;
		int64_t lts = 0;
		bool have_last = t.slam_rels.get_latest(&lts, &lr);
		const bool have_last_orientation = have_last &&
		    (lr.relation_flags & XRT_SPACE_RELATION_ORIENTATION_VALID_BIT) != 0;
		xrt_quat lrot = lr.pose.orientation;

		double dt = time_ns_to_s(nts - lts);

		SLAM_TRACE("Dequeued SLAM pose ts=%ld p=[%f,%f,%f] r=[%f,%f,%f,%f]", //
		           nts, data.px, data.py, data.pz, data.ox, data.oy, data.oz, data.ow);

		// Compute new relation based on new pose and velocities since last pose
		xrt_space_relation rel{};
		rel.relation_flags = XRT_SPACE_RELATION_BITMASK_ALL;
		rel.pose = {nrot, npos};
		rel.linear_velocity = nvel;
		if (!no_visual_observations && have_last_orientation && dt > 0.0) {
			math_quat_finite_difference(&lrot, &nrot, dt, &rel.angular_velocity);
		} else {
			rel.relation_flags = (xrt_space_relation_flags)(rel.relation_flags & ~XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT);
		}

		// A relocalization/reset step turns this finite difference into an angular-velocity
		// artifact (worst recorded event: 59.3 deg over 66.8 ms = a ~890 dps "velocity" for one
		// sample) that consumers would extrapolate by. Clamp it to the IMU-envelope-consistent
		// rate when the step's rotation exceeds the envelope (the same innovation test the world
		// re-anchor guard uses); the true post-step angular velocity is re-established by the
		// next sample pair.
		if (!no_visual_observations && have_last_orientation && dt > 0.0) {
			double gyro_int_deg = 0.0;
			os_mutex_lock(&t.lock_ff);
			uint64_t seg_end_ns = (uint64_t)nts;
			struct xrt_vec3 g;
			uint64_t g_ts;
			for (size_t i = 0; m_ff_vec3_f32_get(t.gyro_ff, i, &g, &g_ts); i++) {
				if ((int64_t)g_ts >= (int64_t)seg_end_ns) {
					continue; // not yet inside [lts, nts]
				}
				int64_t seg_start = (int64_t)g_ts > lts ? (int64_t)g_ts : lts;
				gyro_int_deg +=
				    (double)m_vec3_len(g) * (180.0 / M_PI) * time_ns_to_s(seg_end_ns - seg_start);
				if ((int64_t)g_ts <= lts) {
					break;
				}
				seg_end_ns = (uint64_t)g_ts;
			}
			os_mutex_unlock(&t.lock_ff);

			struct u_world_reanchor_step st;
			if (u_world_reanchor_compute_excess(&u_world_reanchor_default_params, &lr.pose, &rel.pose,
			                                    dt, gyro_int_deg, &st) &&
			    st.exc_ang_deg > 0.0) {
				float av_len = m_vec3_len(rel.angular_velocity);
				double env_rate = (st.dang_deg - st.exc_ang_deg) * (M_PI / 180.0) / dt;
				if (av_len > 0.0f && (double)av_len > env_rate) {
					rel.angular_velocity =
					    m_vec3_mul_scalar(rel.angular_velocity, (float)(env_rate / av_len));
					SLAM_INFO("Clamped re-anchor angular-velocity artifact %.0f -> %.0f dps "
					          "(step %.2f deg, envelope %.2f deg over %.1f ms)",
					          (double)av_len * 180.0 / M_PI, env_rate * 180.0 / M_PI, st.dang_deg,
					          st.dang_deg - st.exc_ang_deg, dt * 1e3);
				}
			}
		}

		if (no_visual_observations) {
			// A pose without any current-frame landmark observation is not an observed
			// room pose. Keep raw diagnostics below, but never publish its inertial drift.
			rel = {};
			rel.pose.orientation.w = 1;
		}

		// Push to relationship history unless we are debugging prediction
		if (no_visual_observations || t.dbg_pred_counter % t.dbg_pred_every == 0) {
			if (t.slam_rels.push(rel, nts)) {
				t.latest_gyro_bias = no_visual_observations ? gyro_bias_sample{nts, false, {}}
				                                          : read_pose_gyro_bias(t, pose, nts);
				t.gyro_bias_writer->push(t.latest_gyro_bias);
			}
		}
		t.dbg_pred_counter = (t.dbg_pred_counter + 1) % t.dbg_pred_every;

		gt_ui_push(t, nts, rel.pose);
		t.slam_traj_writer->push({nts, {nrot, npos}});
		xrt_pose_sample pose_sample = {nts, {nrot, npos}};
		xrt_sink_push_pose(t.euroc_recorder->gt, &pose_sample);

		auto tss = timing_ui_push(t, pose, nts);
		t.slam_times_writer->push(tss);

		if (t.features.enabled) {
			vector feat_count = features_ui_push(t, pose, nts);
			t.slam_features_writer->push({nts, feat_count});
		}

		t.vit.pose_destroy(pose);
	} while (t.vit.tracker_pop_pose(t.tracker, &pose) == VIT_SUCCESS && pose);

	return true;
}

//! Camera delivery may drain independently of device queries. VIT specifies one queue consumer;
//! retain the mutex through relation publication so a later popped pose cannot overtake an earlier one.
static bool
flush_poses(TrackerSlam &t)
{
	unique_lock lock(t.pose_mutex);
	return flush_poses_locked(t);
}

//! Return our best guess of the relation at time @p when_ns using all the data the tracker has.
static void
predict_pose(TrackerSlam &t, timepoint_ns when_ns, struct xrt_space_relation *out_relation)
{
	XRT_TRACE_MARKER();

	bool valid_pred_type = t.pred_type >= SLAM_PRED_NONE && t.pred_type < SLAM_PRED_COUNT;
	SLAM_DASSERT(valid_pred_type, "Invalid prediction type (%d)", t.pred_type);

	// Get last relation computed purely from SLAM data
	xrt_space_relation rel{};
	int64_t rel_ts;
	bool empty = !t.slam_rels.get_latest(&rel_ts, &rel);

	// Stop if there is no previous relation to use for prediction
	if (empty) {
		out_relation->relation_flags = XRT_SPACE_RELATION_BITMASK_NONE;
		return;
	}

	// Use only last SLAM pose without prediction if PREDICTION_NONE
	if (t.pred_type == SLAM_PRED_NONE) {
		*out_relation = rel;
		return;
	}

	// Use only SLAM data if asking for an old point in time or PREDICTION_SP_SO_SA_SL
	SLAM_DASSERT_(rel_ts < INT64_MAX);
	if (t.pred_type == SLAM_PRED_POSE_ONLY || when_ns <= (int64_t)rel_ts) {
		t.slam_rels.get(when_ns, out_relation);
		return;
	}


	// Unobserved future poses must not acquire validity from later IMU integration.
	// Historical queries above retain the validity of the pose at their own time.
	if (rel.relation_flags == XRT_SPACE_RELATION_BITMASK_NONE) {
		*out_relation = rel;
		return;
	}
	const xrt_vec3 *gyro_bias = t.use_pose_gyro_bias && t.latest_gyro_bias.valid &&
	                                    t.latest_gyro_bias.ts == rel_ts
	                                ? &t.latest_gyro_bias.bias
	                                : nullptr;

	if (t.pred_type == SLAM_PRED_DEAD_RECKONING) {
		os_mutex_lock(&t.lock_ff);

		t_apply_dead_reckoning_with_gyro_bias( //
		    t.gyro_ff,             //
		    t.accel_ff,            //
		    &t.gravity_correction, //
		    gyro_bias,             //
		    when_ns,               //
		    &rel,                  //
		    (int64_t)rel_ts,       //
		    out_relation);         //

		os_mutex_unlock(&t.lock_ff);
		return;
	}

	os_mutex_lock(&t.lock_ff);

	// Update angular velocity with gyro data
	if (t.pred_type >= SLAM_PRED_GYRO) {
		xrt_vec3 avg_gyro{};
		m_ff_vec3_f32_filter(t.gyro_ff, rel_ts, when_ns, &avg_gyro);
		if (gyro_bias != nullptr) {
			avg_gyro -= *gyro_bias;
		}
		math_quat_rotate_derivative(&rel.pose.orientation, &avg_gyro, &rel.angular_velocity);
	}

	// Update linear velocity with accel data
	if (t.pred_type >= SLAM_PRED_ACCEL_GYRO) {
		xrt_vec3 avg_accel{};
		m_ff_vec3_f32_filter(t.accel_ff, rel_ts, when_ns, &avg_accel);
		xrt_vec3 world_accel{};
		math_quat_rotate_vec3(&rel.pose.orientation, &avg_accel, &world_accel);
		world_accel += t.gravity_correction;
		double slam_to_imu_dt = time_ns_to_s(t.last_imu_ts - rel_ts);
		rel.linear_velocity += world_accel * slam_to_imu_dt;
	}

	os_mutex_unlock(&t.lock_ff);

	// Do the prediction based on the updated relation
	double slam_to_now_dt = time_ns_to_s(when_ns - rel_ts);
	xrt_space_relation predicted_relation{};
	m_predict_relation(&rel, slam_to_now_dt, &predicted_relation);

	*out_relation = predicted_relation;
}

//! Various filters to remove noise from the predicted trajectory.
static void
filter_pose(TrackerSlam &t, timepoint_ns when_ns, struct xrt_space_relation *out_relation)
{
	XRT_TRACE_MARKER();

	if (t.filter.use_moving_average_filter) {
		if (out_relation->relation_flags & XRT_SPACE_RELATION_POSITION_VALID_BIT) {
			xrt_vec3 pos = out_relation->pose.position;
			m_ff_vec3_f32_push(t.filter.pos_ff, &pos, when_ns);
		}

		if (out_relation->relation_flags & XRT_SPACE_RELATION_ORIENTATION_VALID_BIT) {
			// Don't save w component as we can retrieve it knowing these are (almost) unit quaternions
			xrt_vec3 rot = {out_relation->pose.orientation.x, out_relation->pose.orientation.y,
			                out_relation->pose.orientation.z};
			m_ff_vec3_f32_push(t.filter.rot_ff, &rot, when_ns);
		}

		// Get averages in time window
		timepoint_ns window = t.filter.window * U_TIME_1MS_IN_NS;
		xrt_vec3 avg_pos;
		m_ff_vec3_f32_filter(t.filter.pos_ff, when_ns - window, when_ns, &avg_pos);
		xrt_vec3 avg_rot; // Naive but good enough rotation average
		m_ff_vec3_f32_filter(t.filter.rot_ff, when_ns - window, when_ns, &avg_rot);

		// Considering the naive averaging this W is a bit wrong, but it feels reasonably well
		float avg_rot_w = sqrtf(1 - (avg_rot.x * avg_rot.x + avg_rot.y * avg_rot.y + avg_rot.z * avg_rot.z));
		out_relation->pose.orientation = xrt_quat{avg_rot.x, avg_rot.y, avg_rot.z, avg_rot_w};
		out_relation->pose.position = avg_pos;

		//! @todo Implement the quaternion averaging with a m_ff_vec4_f32 and
		//! normalization. Although it would be best to have a way of generalizing
		//! types before so as to not have redundant copies of ff logic.
	}

	if (t.filter.use_exponential_smoothing_filter) {
		xrt_space_relation &last = t.filter.last;
		xrt_space_relation &target = t.filter.target;
		target = *out_relation;
		m_space_relation_interpolate(&last, &target, t.filter.alpha, target.relation_flags, &last);
		*out_relation = last;
	}

	if (t.filter.use_one_euro_filter) {
		xrt_pose &p = out_relation->pose;
		if (out_relation->relation_flags & XRT_SPACE_RELATION_POSITION_VALID_BIT) {
			m_filter_euro_vec3_run(&t.filter.pos_oe, when_ns, &p.position, &p.position);
		}
		if (out_relation->relation_flags & XRT_SPACE_RELATION_ORIENTATION_VALID_BIT) {
			m_filter_euro_quat_run(&t.filter.rot_oe, when_ns, &p.orientation, &p.orientation);
		}
	}
}

static void
setup_ui(TrackerSlam &t)
{
	t.pred_combo.count = SLAM_PRED_COUNT;
	t.pred_combo.options = "NONE\0POSE_ONLY\0GYRO\0ACCEL_GYRO\0DEAD_RECKONING\0";
	t.pred_combo.value = (int *)&t.pred_type;
	t.ui_sink = vector<u_sink_debug>(t.cam_count);
	for (size_t i = 0; i < t.ui_sink.size(); i++) {
		u_sink_debug_init(&t.ui_sink[i]);
	}
	os_mutex_init(&t.lock_ff);
	m_ff_vec3_f32_alloc(&t.gyro_ff, 1000);
	m_ff_vec3_f32_alloc(&t.accel_ff, 1000);
	m_ff_vec3_f32_alloc(&t.filter.pos_ff, 1000);
	m_ff_vec3_f32_alloc(&t.filter.rot_ff, 1000);

	u_var_add_root(&t, "SLAM Tracker", true);
	u_var_add_log_level(&t, &t.log_level, "Log Level");
	u_var_add_bool(&t, &t.submit, "Submit data to SLAM");

	u_var_button_cb reset_state_cb = [](void *t_ptr) {
		TrackerSlam &t = *(TrackerSlam *)t_ptr;
		unique_lock lock(t.pose_mutex);

		t.latest_gyro_bias = {};
		vit_result_t vres = t.vit.tracker_reset(t.tracker);
		if (vres != VIT_SUCCESS) {
			SLAM_WARN("Failed to reset VIT tracker");
		}
	};
	t.reset_state_btn.cb = reset_state_cb;
	t.reset_state_btn.ptr = &t;
	u_var_add_button(&t, &t.reset_state_btn, "Reset tracker state");

	u_var_add_bool(&t, &t.gt.override_tracking, "Track with ground truth (if available)");
	euroc_recorder_add_ui(t.euroc_recorder, &t, "");

	u_var_add_gui_header(&t, NULL, "Trajectory Filter");
	u_var_add_bool(&t, &t.filter.use_moving_average_filter, "Enable moving average filter");
	u_var_add_f64(&t, &t.filter.window, "Window size (ms)");
	u_var_add_bool(&t, &t.filter.use_exponential_smoothing_filter, "Enable exponential smoothing filter");
	u_var_add_f32(&t, &t.filter.alpha, "Smoothing factor");
	u_var_add_bool(&t, &t.filter.use_one_euro_filter, "Enable one euro filter");
	u_var_add_f32(&t, &t.filter.pos_oe.base.fc_min, "Position minimum cutoff");
	u_var_add_f32(&t, &t.filter.pos_oe.base.beta, "Position beta speed");
	u_var_add_f32(&t, &t.filter.pos_oe.base.fc_min_d, "Position minimum delta cutoff");
	u_var_add_f32(&t, &t.filter.rot_oe.base.fc_min, "Orientation minimum cutoff");
	u_var_add_f32(&t, &t.filter.rot_oe.base.beta, "Orientation beta speed");
	u_var_add_f32(&t, &t.filter.rot_oe.base.fc_min_d, "Orientation minimum delta cutoff");

	u_var_add_gui_header(&t, NULL, "Prediction");
	u_var_add_combo(&t, &t.pred_combo, "Prediction Type");
	u_var_add_i32(&t, &t.dbg_pred_every, "Debug prediction skips (try 30)");
	u_var_add_ro_ff_vec3_f32(&t, t.gyro_ff, "Gyroscope");
	u_var_add_ro_ff_vec3_f32(&t, t.accel_ff, "Accelerometer");
	u_var_add_f32(&t, &t.gravity_correction.z, "Gravity Correction");
	for (size_t i = 0; i < t.ui_sink.size(); i++) {
		char label[64] = {0};
		snprintf(label, sizeof(label), "Camera %zu", i);
		u_var_add_sink_debug(&t, &t.ui_sink[i], label);
	}

	u_var_add_gui_header(&t, NULL, "Stats");
	u_var_add_ro_raw_text(&t, "\nRecord to CSV files", "Record to CSV files");
	u_var_add_bool(&t, &t.slam_traj_writer->enabled, "Record tracked trajectory");
	u_var_add_bool(&t, &t.pred_traj_writer->enabled, "Record predicted trajectory");
	u_var_add_bool(&t, &t.filt_traj_writer->enabled, "Record filtered trajectory");
	u_var_add_bool(&t, &t.slam_times_writer->enabled, "Record tracker times");
	u_var_add_bool(&t, &t.slam_features_writer->enabled, "Record feature count");
	timing_ui_setup(t);
	features_ui_setup(t);
	// Later, gt_ui_setup will setup the tracking error UI if ground truth becomes available
}

static void
add_camera_calibration(const TrackerSlam &t, const t_slam_camera_calibration *calib, uint32_t cam_index)
{
	const t_camera_calibration &view = calib->base;

	vit_camera_calibration params = {};
	params.camera_index = cam_index;
	params.width = view.image_size_pixels.w;
	params.height = view.image_size_pixels.h;
	params.frequency = calib->frequency;

	params.fx = view.intrinsics[0][0];
	params.fy = view.intrinsics[1][1];
	params.cx = view.intrinsics[0][2];
	params.cy = view.intrinsics[1][2];

	switch (view.distortion_model) {
	case T_DISTORTION_OPENCV_RADTAN_8: {
		params.model = VIT_CAMERA_DISTORTION_RT8;
		const size_t size = sizeof(struct t_camera_calibration_rt8_params) + sizeof(double);
		params.distortion_count = size / sizeof(double);
		SLAM_ASSERT_(params.distortion_count == 9);

		memcpy(params.distortion, &view.rt8, size);

		// -1 metric radius tells Basalt to estimate the metric radius on its own.
		params.distortion[8] = -1.f;
		break;
	}
	case T_DISTORTION_WMR: {
		params.model = VIT_CAMERA_DISTORTION_RT8;
		const size_t size = sizeof(struct t_camera_calibration_rt8_params) + sizeof(double);
		params.distortion_count = size / sizeof(double);
		SLAM_ASSERT_(params.distortion_count == 9);

		memcpy(params.distortion, &view.wmr, size);

		params.distortion[8] = view.wmr.rpmax;

		break;
	}
	case T_DISTORTION_FISHEYE_KB4: {
		params.model = VIT_CAMERA_DISTORTION_KB4;
		const size_t size = sizeof(struct t_camera_calibration_kb4_params);
		params.distortion_count = size / sizeof(double);
		SLAM_ASSERT_(params.distortion_count == 4);

		memcpy(params.distortion, &view.kb4, size);
		break;
	}
	default:
		SLAM_ASSERT(false, "SLAM doesn't support distortion type %s",
		            t_stringify_camera_distortion_model(view.distortion_model));
		break;
	}

	xrt_matrix_4x4 T; // Row major T_imu_cam
	math_matrix_4x4_transpose(&calib->T_imu_cam, &T);

	// Converts the xrt_matrix_4x4 from float to double
	for (size_t i = 0; i < ARRAY_SIZE(params.transform); ++i)
		params.transform[i] = T.v[i];

	vit_result_t vres = t.vit.tracker_add_camera_calibration(t.tracker, &params);
	if (vres != VIT_SUCCESS) {
		SLAM_ERROR("Failed to add camera calibration for camera %u", cam_index);
	}
}

static void
add_imu_calibration(TrackerSlam &t, const t_slam_imu_calibration *imu_calib)
{
	vit_imu_calibration_t params = {};
	params.imu_index = 0;
	params.frequency = imu_calib->frequency;

	// TODO improve memcpy size calculation

	const t_inertial_calibration &accel = imu_calib->base.accel;
	memcpy(params.accel.transform, accel.transform, sizeof(double) * 9);
	memcpy(params.accel.offset, accel.offset, sizeof(double) * 3);
	memcpy(params.accel.bias_std, accel.bias_std, sizeof(double) * 3);
	memcpy(params.accel.noise_std, accel.noise_std, sizeof(double) * 3);

	const t_inertial_calibration &gyro = imu_calib->base.gyro;
	memcpy(params.gyro.transform, gyro.transform, sizeof(double) * 9);
	memcpy(params.gyro.offset, gyro.offset, sizeof(double) * 3);
	memcpy(params.gyro.bias_std, gyro.bias_std, sizeof(double) * 3);
	memcpy(params.gyro.noise_std, gyro.noise_std, sizeof(double) * 3);

	vit_result_t vres = t.vit.tracker_add_imu_calibration(t.tracker, &params);
	if (vres != VIT_SUCCESS) {
		SLAM_ERROR("Failed to add imu calibration");
	} else {
		t.prediction_imu_calib = imu_calib->base;
		t.have_prediction_imu_calib = true;
	}
}

static void
send_calibration(TrackerSlam &t, const t_slam_calibration &c)
{
	// Try to send camera calibration data to the SLAM system
	if (t.exts.has_add_camera_calibration) {
		for (int i = 0; i < c.cam_count; i++) {
			SLAM_INFO("Sending Camera %d calibration from Monado", i);
			add_camera_calibration(t, &c.cams[i], i);
		}
	} else {
		SLAM_WARN("Tracker doesn't support camera calibration");
	}

	// Try to send IMU calibration data to the SLAM system
	if (t.exts.has_add_imu_calibration) {
		SLAM_INFO("Sending IMU calibration from Monado");
		add_imu_calibration(t, &c.imu);
	} else {
		SLAM_WARN("Tracker doesn't support IMU calibration");
	}
}

} // namespace xrt::auxiliary::tracking::slam

using namespace xrt::auxiliary::tracking::slam;

/*
 *
 * External functions
 *
 */

//! Get a filtered prediction from the SLAM tracked poses.
extern "C" void
t_slam_get_tracked_pose(struct xrt_tracked_slam *xts, timepoint_ns when_ns, struct xrt_space_relation *out_relation)
{
	XRT_TRACE_MARKER();

	auto &t = *container_of(xts, TrackerSlam, base);

	// One transaction covers draining, query, optional output filters and publication. Even an
	// equal-time query must see newly arrived VIT/IMU data; timestamp-only caching was stale and racy.
	unique_lock lock(t.pose_mutex);
	*out_relation = XRT_SPACE_RELATION_ZERO;
	out_relation->pose.orientation.w = 1;
	flush_poses_locked(t);

	predict_pose(t, when_ns, out_relation);
	t.pred_traj_writer->push({when_ns, out_relation->pose});

	filter_pose(t, when_ns, out_relation);
	t.filt_traj_writer->push({when_ns, out_relation->pose});


	if (t.gt.override_tracking) {
		out_relation->pose = gt2xr_pose(t.gt.origin, get_gt_pose_at(*t.gt.trajectory, when_ns));
	}
}

//! Receive and register ground truth to use for trajectory error metrics.
extern "C" void
t_slam_gt_sink_push(struct xrt_pose_sink *sink, xrt_pose_sample *sample)
{
	XRT_TRACE_MARKER();

	auto &t = *container_of(sink, TrackerSlam, gt_sink);

	if (t.gt.trajectory->empty()) {
		t.gt.origin = sample->pose;
		gt_ui_setup(t);
	}

	t.gt.trajectory->insert_or_assign(sample->timestamp_ns, sample->pose);
	xrt_sink_push_pose(t.euroc_recorder->gt, sample);
}

//! Receive and register masks to use in the next image
extern "C" void
t_slam_hand_mask_sink_push(struct xrt_device_masks_sink *sink, struct xrt_device_masks_sample *hand_masks)
{
	XRT_TRACE_MARKER();

	auto &t = *container_of(sink, TrackerSlam, hand_masks_sink);
	unique_lock lock(t.last_hand_masks_mutex);
	t.last_hand_masks = *hand_masks;
}

//! Receive and register masks to use in the next image
extern "C" void
t_slam_controller_mask_sink_push(struct xrt_device_masks_sink *sink, struct xrt_device_masks_sample *controller_masks)
{
	XRT_TRACE_MARKER();

	auto &t = *container_of(sink, TrackerSlam, controller_masks_sink);
	unique_lock lock(t.last_controller_masks_mutex);
	t.last_controller_masks = *controller_masks;
}

//! Apply driver calibration only to the local prediction copy of a raw sample.
static xrt_vec3
calibrate_prediction_sample(const t_inertial_calibration &calib, const xrt_vec3_f64 &raw)
{
	// Basalt's VIT implementation uses M * raw - offset, with the offset outside M.
	// The WMR driver supplies -factory_bias as offset. Keep all factory matrix terms.
	const double input[3] = {raw.x, raw.y, raw.z};
	double output[3];
	for (int row = 0; row < 3; row++) {
		output[row] = -calib.offset[row];
		for (int col = 0; col < 3; col++) {
			output[row] += calib.transform[row][col] * input[col];
		}
	}
	return {(float)output[0], (float)output[1], (float)output[2]};
}

//! Receive and send IMU samples to the external SLAM system
extern "C" void
t_slam_receive_imu(struct xrt_imu_sink *sink, struct xrt_imu_sample *s)
{
	XRT_TRACE_MARKER();

	auto &t = *container_of(sink, TrackerSlam, imu_sink);

	timepoint_ns ts = s->timestamp_ns;
	xrt_vec3_f64 a = s->accel_m_s2;
	xrt_vec3_f64 w = s->gyro_rad_secs;

	timepoint_ns now = (timepoint_ns)os_monotonic_get_ns();
	SLAM_TRACE("[%ld] imu t=%ld  a=[%f,%f,%f] w=[%f,%f,%f]", now, ts, a.x, a.y, a.z, w.x, w.y, w.z);
	// Check monotonically increasing timestamps
	if (ts <= t.last_imu_ts) {
		SLAM_WARN("Sample (%" PRId64 ") is older than last (%" PRId64 ") by %" PRId64 " ns", ts, t.last_imu_ts,
		          t.last_imu_ts - ts);
		return;
	}
	t.last_imu_ts = ts;

	//! @todo There are many conversions like these between xrt and
	//! slam_tracker.hpp types. Implement a casting mechanism to avoid copies.
	vit_imu_sample_t sample = {};
	sample.timestamp = ts;
	sample.ax = a.x;
	sample.ay = a.y;
	sample.az = a.z;
	sample.wx = w.x;
	sample.wy = w.y;
	sample.wz = w.z;

	if (t.submit) {
		t.vit.tracker_push_imu_sample(t.tracker, &sample);
	}

	xrt_sink_push_imu(t.euroc_recorder->imu, s);

	struct xrt_vec3 gyro = {(float)w.x, (float)w.y, (float)w.z};
	struct xrt_vec3 accel = {(float)a.x, (float)a.y, (float)a.z};
	// VIT and the recorder above still receive raw samples. Prediction needs calibrated
	// samples, since its pose and velocity came from the calibrated tracker estimate.
	if (t.have_prediction_imu_calib) {
		gyro = calibrate_prediction_sample(t.prediction_imu_calib.gyro, w);
		accel = calibrate_prediction_sample(t.prediction_imu_calib.accel, a);
	}
	os_mutex_lock(&t.lock_ff);
	m_ff_vec3_f32_push(t.gyro_ff, &gyro, ts);
	m_ff_vec3_f32_push(t.accel_ff, &accel, ts);
	os_mutex_unlock(&t.lock_ff);
}

//! Push the frame to the external SLAM system
static void
receive_frame(TrackerSlam &t, struct xrt_frame *frame, uint32_t cam_index)
{
	XRT_TRACE_MARKER();

	// Return early if we don't submit
	if (!t.submit) {
		return;
	}

	if (cam_index == t.cam_count - 1) {
		flush_poses(t); // Useful to flush SLAM poses when no openxr app is open
	}

	SLAM_DASSERT(t.cam_ts_guard[0].has_pushed || cam_index == 0, "First frame was not a cam0 frame");

	// Enforce a strictly-monotonic, plausible timeline per camera BEFORE anything reaches the
	// SLAM system: the WMR device->monotonic conversion offset wobbles under USB churn and can
	// regress converted group timestamps by tens of ms (2026-07-06 launch-window crash class),
	// and Basalt turns any regression into an abort() inside the calling process (vrserver).
	timepoint_ns ts = (int64_t)frame->timestamp;
	SLAM_TRACE("[%" PRId64 "] cam%d frame t=%" PRId64, os_monotonic_get_ns(), cam_index, ts);
	timepoint_ns prev_pushed = t.cam_ts_guard[cam_index].last_pushed_ns;
	enum u_frame_ts_guard_verdict verdict = u_frame_ts_guard_check(&t.cam_ts_guard[cam_index], ts);
	if (verdict != U_FRAME_TS_GUARD_ACCEPT) {
		t.ts_drop_count++;
		g2_telem_event(0, (uint64_t)ts, G2_TELEM_EV_SLAM_FRAME_TS_DROPPED,
		               (float)((double)(ts - prev_pushed) / (double)U_TIME_1MS_IN_NS));
		timepoint_ns now = os_monotonic_get_ns();
		if (now - t.last_ts_drop_warn_mono >= (timepoint_ns)U_TIME_1S_IN_NS) {
			t.last_ts_drop_warn_mono = now;
			SLAM_WARN("Dropping cam%d frame (t=%" PRId64 "): %s vs last pushed %" PRId64
			          " (delta %" PRId64 " ns); %" PRIu64 " frame(s) dropped so far",
			          cam_index, ts,
			          verdict == U_FRAME_TS_GUARD_DROP_REGRESSED ? "timestamp regressed"
			                                                     : "implausible forward jump",
			          prev_pushed, ts - prev_pushed, t.ts_drop_count);
		}
		return;
	}

	// Construct and send the image sample
	vit_img_sample sample = {};
	sample.cam_index = cam_index;
	sample.timestamp = ts;

	sample.data = frame->data;
	sample.width = frame->width;
	sample.height = frame->height;
	sample.stride = frame->stride;
	sample.size = frame->size;

	// TODO check format before
	switch (frame->format) {
	case XRT_FORMAT_L8: sample.format = VIT_IMAGE_FORMAT_L8; break;
	case XRT_FORMAT_R8G8B8: sample.format = VIT_IMAGE_FORMAT_R8G8B8; break;
	default: SLAM_ERROR("Unknown image format"); return;
	}

	xrt_device_masks_sample mask_samples[2];
	{
		unique_lock hand_masks_lock(t.last_hand_masks_mutex);
		mask_samples[0] = t.last_hand_masks;

		unique_lock controller_masks_lock(t.last_controller_masks_mutex);
		mask_samples[1] = t.last_controller_masks;
	}

	std::vector<vit_mask_t> masks;

	for (auto &mask_sample : mask_samples) {
		auto &device_view = mask_sample.views[cam_index];
		for (auto &hand : device_view.devices) {
			if (!hand.enabled) {
				continue;
			}

			vit_mask_t mask{};
			mask.x = hand.rect.x;
			mask.y = hand.rect.y;
			mask.w = hand.rect.w;
			mask.h = hand.rect.h;
			masks.push_back(mask);
		}
	}

	sample.mask_count = masks.size();
	sample.masks = masks.empty() ? nullptr : masks.data();

	{
		XRT_TRACE_IDENT(slam_push);
		t.vit.tracker_push_img_sample(t.tracker, &sample);
	}
}

#define DEFINE_RECEIVE_CAM(cam_id)                                                                                     \
	extern "C" void t_slam_receive_cam##cam_id(struct xrt_frame_sink *sink, struct xrt_frame *frame)               \
	{                                                                                                              \
		auto &t = *container_of(sink, TrackerSlam, cam_sinks[cam_id]);                                         \
		receive_frame(t, frame, cam_id);                                                                       \
		u_sink_debug_push_frame(&t.ui_sink[cam_id], frame);                                                    \
		xrt_sink_push_frame(t.euroc_recorder->cams[cam_id], frame);                                            \
	}

DEFINE_RECEIVE_CAM(0)
DEFINE_RECEIVE_CAM(1)
DEFINE_RECEIVE_CAM(2)
DEFINE_RECEIVE_CAM(3)
DEFINE_RECEIVE_CAM(4)

//! Define a function for each XRT_TRACKING_MAX_SLAM_CAMS and reference it in this array
void (*t_slam_receive_cam[XRT_TRACKING_MAX_SLAM_CAMS])(xrt_frame_sink *, xrt_frame *) = {
    t_slam_receive_cam0, //
    t_slam_receive_cam1, //
    t_slam_receive_cam2, //
    t_slam_receive_cam3, //
    t_slam_receive_cam4, //
};


extern "C" void
t_slam_node_break_apart(struct xrt_frame_node *node)
{
	auto &t = *container_of(node, TrackerSlam, node);
	if (t.ovr_tracker != NULL) {
		t_openvr_tracker_stop(t.ovr_tracker);
	}

	vit_result_t vres = t.vit.tracker_stop(t.tracker);
	if (vres != VIT_SUCCESS) {
		SLAM_ERROR("Failed to stop VIT tracker");
		return;
	}

	SLAM_DEBUG("SLAM tracker dismantled");
}

extern "C" void
t_slam_node_destroy(struct xrt_frame_node *node)
{
	auto t_ptr = container_of(node, TrackerSlam, node);
	auto &t = *t_ptr; // Needed by SLAM_DEBUG
	SLAM_DEBUG("Destroying SLAM tracker");
	if (t.ovr_tracker != NULL) {
		t_openvr_tracker_destroy(t.ovr_tracker);
	}
	delete t.gt.trajectory;
	delete t.slam_times_writer;
	delete t.slam_features_writer;
	delete t.gyro_bias_writer;
	delete t.slam_traj_writer;
	delete t.pred_traj_writer;
	delete t.filt_traj_writer;
	u_var_remove_root(t_ptr);
	for (size_t i = 0; i < t.ui_sink.size(); i++) {
		u_sink_debug_destroy(&t.ui_sink[i]);
	}
	m_ff_vec3_f32_free(&t.gyro_ff);
	m_ff_vec3_f32_free(&t.accel_ff);
	os_mutex_destroy(&t.lock_ff);
	m_ff_vec3_f32_free(&t.filter.pos_ff);
	m_ff_vec3_f32_free(&t.filter.rot_ff);

	t_ptr->vit.tracker_destroy(t_ptr->tracker);
	t_vit_bundle_unload(&t_ptr->vit);

	/* Telemetry shutdown is NOT done here: this SLAM frame node is destroyed
	 * before the other telemetry producers (controllers, WMR source), so flushing
	 * here could race a still-running producer. Ownership moved to wmr_hmd_destroy(),
	 * which runs after all producers have stopped (BLOCKER-1 / MAJOR-4). */

	delete t_ptr;
}

extern "C" int
t_slam_start(struct xrt_tracked_slam *xts)
{
	auto &t = *container_of(xts, TrackerSlam, base);
	vit_result_t vres = t.vit.tracker_start(t.tracker);
	if (vres != VIT_SUCCESS) {
		SLAM_ERROR("Failed to start VIT tracker");
		return -1;
	}

	SLAM_DEBUG("SLAM tracker started");
	return 0;
}

extern "C" void
t_slam_fill_default_config(struct t_slam_tracker_config *config)
{
	config->log_level = debug_get_log_option_slam_log();
	config->vit_system_library_path = debug_get_option_vit_system_library_path();
	config->slam_config = debug_get_option_slam_config();
	config->slam_ui = debug_get_bool_option_slam_ui();
	config->submit_from_start = debug_get_bool_option_slam_submit_from_start();
	config->openvr_groundtruth_device = int(debug_get_num_option_slam_openvr_groundtruth_device());
	config->prediction = t_slam_prediction_type(debug_get_num_option_slam_prediction_type());
	config->write_csvs = debug_get_bool_option_slam_write_csvs();
	config->csv_path = debug_get_option_slam_csv_path();
	config->timing_stat = debug_get_bool_option_slam_timing_stat();
	config->features_stat = debug_get_bool_option_slam_features_stat();
	config->cam_count = int(debug_get_num_option_slam_cam_count());
	config->slam_calib = NULL;
}

extern "C" int
t_slam_create(struct xrt_frame_context *xfctx,
              struct t_slam_tracker_config *config,
              struct xrt_tracked_slam **out_xts,
              struct xrt_slam_sinks **out_sink)
{
	struct t_slam_tracker_config default_config = {};
	if (config == nullptr) {
		t_slam_fill_default_config(&default_config);
		config = &default_config;
	}

	enum u_logging_level log_level = config->log_level;

	std::unique_ptr<TrackerSlam> t_ptr = std::make_unique<TrackerSlam>();
	TrackerSlam &t = *t_ptr;

	t.log_level = log_level;

	SLAM_INFO("Loading VIT system library from VIT_SYSTEM_LIBRARY_PATH='%s'", config->vit_system_library_path);

	if (!t_vit_bundle_load(&t.vit, config->vit_system_library_path)) {
		SLAM_ERROR("Failed to load VIT system library from '%s'", config->vit_system_library_path);
		return -1;
	}

	// Check the user has provided a SLAM_CONFIG file
	const char *config_file = config->slam_config;
	bool some_calib = config->slam_calib != nullptr;
	if (!config_file && !some_calib) {
		SLAM_WARN("Unable to determine sensor calibration, did you forget to set SLAM_CONFIG?");
		return -1;
	}

	struct vit_config system_config = {};
	system_config.file = config_file;
	system_config.cam_count = config->cam_count;
	system_config.show_ui = config->slam_ui;

	vit_result_t vres = t.vit.tracker_create(&system_config, &t.tracker);
	if (vres != VIT_SUCCESS) {
		SLAM_ERROR("Failed to create VIT tracker (%d)", vres);
		return -1;
	}

	vres = t.vit.tracker_get_supported_extensions(t.tracker, &t.exts);
	if (vres != VIT_SUCCESS) {
		SLAM_ERROR("Failed to get VIT tracker supported extensions (%d)", vres);
		return -1;
	}

	t.base.get_tracked_pose = t_slam_get_tracked_pose;

	if (!config_file) {
		SLAM_INFO("Using calibration from driver and default pipeline settings");
		send_calibration(t, *config->slam_calib); // Not null because of `some_calib`
	} else {
		SLAM_INFO("Using sensor calibration provided by the SLAM_CONFIG file");
	}

	SLAM_ASSERT(t_slam_receive_cam[ARRAY_SIZE(t_slam_receive_cam) - 1] != nullptr, "See `cam_sink_push` docs");
	t.sinks.cam_count = config->cam_count;
	for (int i = 0; i < XRT_TRACKING_MAX_SLAM_CAMS; i++) {
		t.cam_sinks[i].push_frame = t_slam_receive_cam[i];
		t.sinks.cams[i] = &t.cam_sinks[i];
	}

	t.imu_sink.push_imu = t_slam_receive_imu;
	t.sinks.imu = &t.imu_sink;

	t.gt_sink.push_pose = t_slam_gt_sink_push;
	t.sinks.gt = &t.gt_sink;

	t.hand_masks_sink.push_device_masks = t_slam_hand_mask_sink_push;
	t.sinks.hand_masks = &t.hand_masks_sink;

	t.controller_masks_sink.push_device_masks = t_slam_controller_mask_sink_push;
	t.sinks.controller_masks = &t.controller_masks_sink;

	t.submit = config->submit_from_start;
	t.cam_count = config->cam_count;

	t.node.break_apart = t_slam_node_break_apart;
	t.node.destroy = t_slam_node_destroy;

	xrt_frame_context_add(xfctx, &t.node);

	// Auto-start raw EuRoC recording when G2_RECORD=<dir> is set in the environment.
	// Uses the same record-from-start path the debug GUI button drives, recording on the
	// same monotonic timebase as the telemetry/frames. Otherwise behaves as before.
	const char *g2_record_dir = getenv("G2_RECORD");
	bool g2_record = g2_record_dir != NULL && g2_record_dir[0] != '\0';
	if (g2_record) {
		SLAM_INFO("G2_RECORD set: auto-recording EuRoC dataset to '%s'", g2_record_dir);
	}
	t.euroc_recorder = euroc_recorder_create(xfctx, g2_record ? g2_record_dir : NULL, t.cam_count, g2_record);

	// Bring up telemetry if G2_TELEMETRY=<dir> is set (no-op otherwise). Idempotent:
	// wmr_hmd_create() already calls this; only the first caller wins. Kept here so a
	// standalone/non-WMR SLAM path still gets telemetry. Shutdown is owned by
	// wmr_hmd_destroy() (runs after all producers stop -- BLOCKER-1 / MAJOR-4).
	g2_telem_init(getenv("G2_TELEMETRY"));

	t.last_imu_ts = INT64_MIN;
	for (struct u_frame_ts_guard &guard : t.cam_ts_guard) {
		u_frame_ts_guard_init(&guard);
	}
	t.last_ts_drop_warn_mono = 0; // not INT64_MIN: `now - limiter` must stay overflow-free
	t.ts_drop_count = 0;
	t.last_hand_masks = xrt_device_masks_sample{};
	t.last_controller_masks = xrt_device_masks_sample{};

	t.pred_type = config->prediction;

	m_filter_euro_vec3_init(&t.filter.pos_oe, t.filter.min_cutoff, t.filter.min_dcutoff, t.filter.beta);
	m_filter_euro_quat_init(&t.filter.rot_oe, t.filter.min_cutoff, t.filter.min_dcutoff, t.filter.beta);

	t.gt.trajectory = new Trajectory{};

	// Setup CSV files
	bool write_csvs = config->write_csvs;
	string dir = config->csv_path;
	t.slam_times_writer = new TimingWriter(dir, "timing.csv", write_csvs, t.timing.columns);
	t.slam_features_writer = new FeaturesWriter(dir, "features.csv", write_csvs, t.cam_count);
	t.gyro_bias_writer = new CSVWriter<gyro_bias_sample>(dir, "gyro-bias.csv", write_csvs,
	    {"timestamp", "valid", "gyro_bias_x_rad_s", "gyro_bias_y_rad_s", "gyro_bias_z_rad_s"});
	t.slam_traj_writer = new TrajectoryWriter(dir, "tracking.csv", write_csvs);
	t.pred_traj_writer = new TrajectoryWriter(dir, "prediction.csv", write_csvs);
	t.filt_traj_writer = new TrajectoryWriter(dir, "filtering.csv", write_csvs);

	setup_ui(t);
	t.use_pose_gyro_bias = debug_get_bool_option_g2_predict_with_vit_bias();
	if (t.use_pose_gyro_bias && !t.have_prediction_imu_calib) {
		SLAM_WARN("Pose gyro bias requested without matching driver IMU calibration; retaining existing prediction");
		t.use_pose_gyro_bias = false;
	}
	if (t.use_pose_gyro_bias) {
		if (t.vit.g2_pose_get_imu_bias_v1 != nullptr) {
			SLAM_INFO("Pose-coupled learned gyro bias enabled for prediction; accelerometer path unchanged");
		} else {
			SLAM_WARN("Pose gyro bias requested but unavailable; retaining existing prediction");
		}
	}

	t.require_visual_observations = debug_get_bool_option_g2_require_visual_observations();
	if (t.require_visual_observations) {
		if (t.exts.has_pose_features &&
		    t.vit.tracker_enable_extension(t.tracker, VIT_TRACKER_EXTENSION_POSE_FEATURES, true) == VIT_SUCCESS) {
			t.visual_observations_available = true;
			t.features.enabled = true;
			// The extension now supplies pose validity, not just the optional debug graph.
			t.features.enable_btn.disabled = true;
			SLAM_INFO("Current-frame visual observation validation enabled");
		} else {
			SLAM_WARN("Visual observation validation requested but unsupported; observation status is unknown");
		}
	}

	// Setup OpenVR groundtruth tracker
	if (config->openvr_groundtruth_device > 0) {
		enum openvr_device dev_class = openvr_device(config->openvr_groundtruth_device);
		const double freq = 1000.0f;
		t.ovr_tracker = t_openvr_tracker_create(freq, &dev_class, &t.sinks.gt, 1);
		if (t.ovr_tracker != NULL) {
			t_openvr_tracker_start(t.ovr_tracker);
		}
	}

	// Get ownership
	TrackerSlam *tracker = t_ptr.release();

	*out_xts = &tracker->base;
	*out_sink = &tracker->sinks;

	SLAM_DEBUG("SLAM tracker created");
	return 0;
}
