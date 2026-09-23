# Copyright 2018-2024, Collabora, Ltd.
# Copyright 2024-2026, NVIDIA CORPORATION.
#
# SPDX-License-Identifier: BSL-1.0

set(tests
    tests_cxx_wrappers
    tests_deque
    tests_frame_ts_guard
    tests_generic_callbacks
    tests_glue_device
    tests_history_buf
    tests_id_ringbuffer
    tests_json
    tests_lowpass_float
    tests_lowpass_integer
    tests_pacing
    tests_quatexpmap
    tests_quat_change_of_basis
    tests_quat_swing_twist
    tests_rational
    tests_relation_chain
	tests_slam_feature_support
    tests_vector
    tests_worker
    tests_pose
    tests_vec3_angle
    tests_visibility_mask
    tests_wmr_camera_footer
    tests_world_reanchor
	)
if(XRT_HAVE_OPENCV)
	list(APPEND tests tests_kalman_fusion)
endif()
if(XRT_HAVE_D3D11)
	list(APPEND tests tests_aux_d3d_d3d11 tests_comp_client_d3d11)
endif()
if(XRT_HAVE_D3D12)
	list(APPEND tests tests_aux_d3d_d3d12 tests_comp_client_d3d12)
endif()
if(XRT_HAVE_VULKAN)
	list(APPEND tests tests_comp_client_vulkan tests_uv_to_tangent)
endif()
if(XRT_FEATURE_OPENXR)
	list(APPEND tests tests_input_transform)
endif()
if(XRT_HAVE_OPENGL
   AND XRT_HAVE_OPENGL_GLX
   AND XRT_HAVE_SDL2
   AND (WIN32 OR VK_USE_PLATFORM_XLIB_XRANDR_EXT)
	)
	set(_have_opengl_test ON)
	list(APPEND tests tests_comp_client_opengl)
endif()
if(XRT_BUILD_DRIVER_HANDTRACKING)
	list(APPEND tests tests_levenbergmarquardt)
endif()

# Every case in these is tagged [.][needgpu], so on a headless machine the binary legitimately runs zero
# cases. Only they may pass --allow-running-no-tests, and ctest turns their "No tests ran" banner into a
# visible SKIP: a green square over zero coverage is indistinguishable from a green square over real
# coverage. Every other test must FAIL if it ever runs nothing.
set(gpu_gated_tests
    tests_aux_d3d_d3d11
    tests_aux_d3d_d3d12
    tests_comp_client_d3d11
    tests_comp_client_d3d12
    tests_comp_client_opengl
    tests_comp_client_vulkan
	)

foreach(testname ${tests})
	add_executable(${testname} ${testname}.cpp)
	target_link_libraries(${testname} PRIVATE xrt-external-catch2)
	target_link_libraries(${testname} PRIVATE aux_util)
	if(testname IN_LIST gpu_gated_tests)
		add_test(NAME ${testname} COMMAND ${testname} --success --allow-running-no-tests)
		set_tests_properties(${testname} PROPERTIES SKIP_REGULAR_EXPRESSION "No tests ran")
	else()
		add_test(NAME ${testname} COMMAND ${testname} --success)
	endif()
	set_target_properties(${testname} PROPERTIES FOLDER monado_tests)
endforeach()

# For tests that require more than just aux_util, link those other libs down here.

target_link_libraries(tests_cxx_wrappers PRIVATE xrt-interfaces)
target_link_libraries(tests_glue_device PRIVATE xrt-interfaces aux_glue)
target_link_libraries(tests_history_buf PRIVATE aux_math)
target_link_libraries(tests_lowpass_float PRIVATE aux_math)
target_link_libraries(tests_lowpass_integer PRIVATE aux_math)
target_link_libraries(tests_quatexpmap PRIVATE aux_math)
target_link_libraries(tests_rational PRIVATE aux_math)
target_link_libraries(tests_relation_chain PRIVATE aux_math)
target_link_libraries(tests_pose PRIVATE aux_math)
target_link_libraries(tests_quat_change_of_basis PRIVATE aux_math)
target_link_libraries(tests_quat_swing_twist PRIVATE aux_math)
target_link_libraries(tests_vec3_angle PRIVATE aux_math)
target_link_libraries(tests_visibility_mask PRIVATE aux_math)
target_include_directories(tests_slam_feature_support PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/external/vit_includes)
# Recorded-head-stream fixture (b2_reanchor_head_streams.bin, generated from the B2 design's
# reference simulator); the test warns + skips the recorded-stream sections if it is absent.
target_compile_definitions(tests_world_reanchor PRIVATE G2_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(tests_association_hypothesis tests_association_hypothesis.cpp)
target_link_libraries(tests_association_hypothesis PRIVATE xrt-external-catch2 aux_util)
target_include_directories(
	tests_association_hypothesis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/tracking/constellation)
add_test(NAME tests_association_hypothesis COMMAND tests_association_hypothesis --success)
set_target_properties(tests_association_hypothesis PROPERTIES FOLDER monado_tests)

add_executable(tests_l2_accept_gate tests_l2_accept_gate.cpp)
target_link_libraries(tests_l2_accept_gate PRIVATE xrt-external-catch2)
target_include_directories(
	tests_l2_accept_gate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/tracking/constellation)
add_test(NAME tests_l2_accept_gate COMMAND tests_l2_accept_gate --success)
set_target_properties(tests_l2_accept_gate PROPERTIES FOLDER monado_tests)

# Recorded cam0 timestamp sequences of the 2026-07-06 crashed + healthy sessions.
target_compile_definitions(tests_frame_ts_guard PRIVATE G2_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# Camera transfer footer validation (header-only parser in the WMR driver).
target_include_directories(tests_wmr_camera_footer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/drivers)

if(XRT_HAVE_OPENCV)
	target_link_libraries(tests_kalman_fusion PRIVATE aux_math aux_tracking Threads::Threads ${OpenCV_LIBRARIES})
	target_include_directories(tests_kalman_fusion PRIVATE ${OpenCV_INCLUDE_DIRS})
	# Path to the committed real-session .replay fixtures; the real-data corpus replay test loads
	# these at runtime and no-ops gracefully if the directory is absent.
	target_compile_definitions(tests_kalman_fusion PRIVATE G2_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()

# Offline controller-VIO replay harness: drives the real constellation tracker + ESKF over recorded raw
# frames + controller IMU (for front-end iteration + IMU<->visual drift study, headless). A tool, not a
# ctest. Needs the constellation tracker (built with the WMR/rift_s constellation drivers) + OpenCV.
if(XRT_HAVE_OPENCV AND TARGET constellation_tracker)
	add_executable(offline_vio_replay offline_vio_replay.cpp)
	target_link_libraries(
		offline_vio_replay
		PRIVATE constellation_tracker aux_tracking aux_util aux_math aux_os xrt-interfaces
				Threads::Threads ${OpenCV_LIBRARIES})
	target_include_directories(offline_vio_replay PRIVATE ${OpenCV_INCLUDE_DIRS}
							  ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/drivers)
	set_target_properties(offline_vio_replay PROPERTIES FOLDER monado_tests)
	add_test(NAME offline_vio_replay_clock_contract COMMAND offline_vio_replay --self-test-clock-contract)

	# Standing regression guardrail: replays the faithful capture through offline_vio_replay and fails if a
	# flip / wrong-branch / yield metric regresses past tolerance vs tests/regression_baseline.json. The
	# runner skips cleanly (exit 0) when the capture, the harness binary, or the scoring deps are absent, so
	# a capture-less / dep-less build never breaks. Prefer the analysis conda env if present, else Python3.
	find_program(
		G2_REGRESSION_PYTHON
		NAMES python
		HINTS "$ENV{HOME}/miniconda3/envs/g2vr/bin"
		NO_DEFAULT_PATH)
	if(NOT G2_REGRESSION_PYTHON)
		set(G2_REGRESSION_PYTHON ${Python3_EXECUTABLE})
	endif()

	# The scoring tools live in the research tree that contains this checkout, which is three levels
	# up from the MAIN checkout and four or more from a git worktree — the project's normal way of
	# working. A fixed ../../../ therefore silently pointed both standing gates at a nonexistent file
	# in every worktree. Walk up until the tools are found; if they are not, keep the fixed path so
	# the gates fail loudly rather than disappearing from the board.
	set(G2_TELEMETRY_TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/telemetry)
	set(_g2_probe ${CMAKE_CURRENT_SOURCE_DIR})
	foreach(_ RANGE 8)
		get_filename_component(_g2_probe ${_g2_probe} DIRECTORY)
		if(EXISTS ${_g2_probe}/tools/telemetry/regression_check.py)
			set(G2_TELEMETRY_TOOLS ${_g2_probe}/tools/telemetry)
			break()
		endif()
	endforeach()
	add_test(
		NAME regression_benchmark
		COMMAND ${G2_REGRESSION_PYTHON}
				${G2_TELEMETRY_TOOLS}/regression_check.py
				--bin $<TARGET_FILE:offline_vio_replay> --baseline
				${CMAKE_CURRENT_SOURCE_DIR}/regression_baseline.json --out
				${CMAKE_CURRENT_BINARY_DIR}/regression_benchmark)
	# --out is build-dir scoped like identity_swap_regression below, NOT the tool's global /tmp default:
	# two checkouts running ctest concurrently otherwise interleave their replay CSVs in one directory,
	# which fails the run at best (observed: csv field-limit error) and scores one binary against the
	# other's rows at worst -- a gate reading someone else's evidence.
	# regression_check.py exits 77 on every skip path (missing scoring deps / capture / binary /
	# controller jsons / baseline), mirroring coast_regression below: a capture-less checkout
	# reports a LOUD ctest SKIP, never a silent pass.
	set_tests_properties(
		regression_benchmark PROPERTIES ENVIRONMENT "PYTHONNOUSERSITE=1" SKIP_RETURN_CODE 77 LABELS
										g2_regression)

	# Standing CROSS-DEVICE IDENTITY gate: replays every pinned identity-swap fixture (the xv1 session
	# whose 300 ms optical blackouts made the two controllers trade LED rings, and the July-23 session
	# whose hands-close instants do it with no forced dropout at all) and fails if ANY accepted optical
	# pose inside a frozen episode window lands on the partner's cleaned-GT instead of its own.
	# Distinct from regression_benchmark, which scores each device against its own reference and so
	# cannot tell a swap from ordinary error. Skips 77 like the gates above when a capture, the
	# harness or the scoring deps are absent.
	add_test(
		NAME identity_swap_regression
		COMMAND ${G2_REGRESSION_PYTHON}
				${G2_TELEMETRY_TOOLS}/identity_swap_check.py
				${CMAKE_CURRENT_BINARY_DIR}/identity_swap_regression --regression --bin
				$<TARGET_FILE:offline_vio_replay>)
	set_tests_properties(
		identity_swap_regression PROPERTIES ENVIRONMENT "PYTHONNOUSERSITE=1" SKIP_RETURN_CODE 77 LABELS
											g2_regression)

	# Inertial coast-drift experiment + GATE: real ESKF on recorded IMU + optical (also with withheld-
	# optical windows) to measure out-of-view dead-reckon drift, the frozen fraction, and the re-entry
	# snap. As a tool it prints the per-gap table; with --check it compares the per-bin metrics against
	# tests/coast_baseline.json and exits nonzero on regression. Registered as the coast_regression
	# ctest below.
	add_executable(eskf_coast_experiment eskf_coast_experiment.cpp)
	target_link_libraries(
		eskf_coast_experiment PRIVATE aux_math aux_tracking xrt-interfaces xrt-external-cjson
									  Threads::Threads ${OpenCV_LIBRARIES})
	set_target_properties(eskf_coast_experiment PROPERTIES FOLDER monado_tests)

	# Standing controller out-of-view / coast regression gate: drives the real ESKF over the committed
	# coast corpus and fails if pos drift / flip% / frozen% / the re-entry snap regress past tolerance vs
	# tests/coast_baseline.json. The binary exits 77 (LOUD ctest SKIP) when a fixture or the baseline is
	# absent, so a fixture-less checkout reports "not run", never a silent pass.
	#
	# The corpus is THREE real recorded sessions x BOTH controllers (captures 20260524-200416-headpose,
	# 20260526-175615-clean and 20260528-080421-xv-session1, extracted by
	# tools/telemetry/make_replay_fixture.py), and the withhold list is the set of coast horizons each leg
	# is replayed at. One session's natural gaps are ~all optical cadence, which left every bin past
	# 100 ms holding a handful of samples -- below the gate's minimum, so the whole out-of-view regime
	# was printed and ignored. The horizons hide the recorded optical stream for that
	# long every --recover seconds, so the filter dead-reckons a real controller through a real
	# out-of-view window and is scored against the real pose that ends it; pooled over the corpus every
	# bin clears the gating minimum. Losing that coverage is itself a gated failure.
	set(G2_COAST_CORPUS
		${CMAKE_CURRENT_SOURCE_DIR}/data/headpose_left.replay
		${CMAKE_CURRENT_SOURCE_DIR}/data/headpose_right.replay
		${CMAKE_CURRENT_SOURCE_DIR}/data/clean_left.replay
		${CMAKE_CURRENT_SOURCE_DIR}/data/clean_right.replay
		${CMAKE_CURRENT_SOURCE_DIR}/data/xv1_left.replay
		${CMAKE_CURRENT_SOURCE_DIR}/data/xv1_right.replay)
	set(G2_COAST_HORIZONS
		--withhold 0 --withhold 0.13 --withhold 0.25 --withhold 0.40 --withhold 0.70 --withhold 1.40
		--withhold 2.50)
	add_test(NAME coast_regression
			 COMMAND eskf_coast_experiment ${G2_COAST_CORPUS} ${G2_COAST_HORIZONS} --check
					 ${CMAKE_CURRENT_SOURCE_DIR}/coast_baseline.json)
	set_tests_properties(coast_regression PROPERTIES SKIP_RETURN_CODE 77 LABELS g2_regression)

	# Decoupled unit tests for the constellation PnP mirror-twin enumeration (synthetic geometry,
	# known truth). A real ctest.
	add_executable(tests_constellation_pnp tests_constellation_pnp.cpp)
	target_link_libraries(
		tests_constellation_pnp PRIVATE xrt-external-catch2 constellation_tracker aux_math aux_util
										xrt-interfaces ${OpenCV_LIBRARIES})
	target_include_directories(
		tests_constellation_pnp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/tracking/constellation
										${OpenCV_INCLUDE_DIRS})
	add_test(NAME tests_constellation_pnp COMMAND tests_constellation_pnp --success)
	set_target_properties(tests_constellation_pnp PROPERTIES FOLDER monado_tests)

	# Decoupled unit tests for per-LED multi-camera triangulation and the L1 single-cam-depth verdict.
	add_executable(tests_multicam_triangulate tests_multicam_triangulate.cpp)
	target_link_libraries(
		tests_multicam_triangulate PRIVATE xrt-external-catch2 constellation_tracker aux_math aux_util
										   xrt-interfaces ${OpenCV_LIBRARIES})
	target_include_directories(
		tests_multicam_triangulate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/tracking/constellation
										   ${OpenCV_INCLUDE_DIRS})
	add_test(NAME tests_multicam_triangulate COMMAND tests_multicam_triangulate --success)
	set_target_properties(tests_multicam_triangulate PROPERTIES FOLDER monado_tests)

	# Decoupled unit tests for the kept-alive 2-mode yaw belief.
	add_executable(tests_yaw_belief tests_yaw_belief.cpp)
	target_link_libraries(
		tests_yaw_belief PRIVATE xrt-external-catch2 constellation_tracker aux_math aux_util
								 xrt-interfaces ${OpenCV_LIBRARIES})
	target_include_directories(
		tests_yaw_belief PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/tracking/constellation
								 ${OpenCV_INCLUDE_DIRS})
	add_test(NAME tests_yaw_belief COMMAND tests_yaw_belief --success)
	set_target_properties(tests_yaw_belief PROPERTIES FOLDER monado_tests)

	# Decoupled, adversarial unit tests for the blob-detection layer (saturated-centroid recovery,
	# per-blob R, reflection rejection). Synthetic frames, known truth.
	add_executable(tests_blobwatch tests_blobwatch.cpp)
	target_link_libraries(
		tests_blobwatch PRIVATE xrt-external-catch2 constellation_tracker aux_math aux_util
								xrt-interfaces ${OpenCV_LIBRARIES})
	target_include_directories(
		tests_blobwatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/tracking/constellation
								${OpenCV_INCLUDE_DIRS})
	add_test(NAME tests_blobwatch COMMAND tests_blobwatch --success)
	set_target_properties(tests_blobwatch PROPERTIES FOLDER monado_tests)

	# Detection-eval harness: runs the REAL blobwatch on PGM frames and dumps blob centroids as
	# JSON, so detector recall/precision can be scored against the annotated LED GT (tools/led_dataset)
	# independently of the matcher. A tool, not a ctest.
	add_executable(blobdump blobdump.cpp)
	target_link_libraries(blobdump PRIVATE constellation_tracker aux_util xrt-interfaces ${OpenCV_LIBRARIES})
	target_include_directories(
		blobdump PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/tracking/constellation
						 ${OpenCV_INCLUDE_DIRS})
	set_target_properties(blobdump PROPERTIES FOLDER monado_tests)
endif()

if(XRT_BUILD_DRIVER_WMR)
	# Regression test for WMR controller IMU calibration selection: the first (ICM-20602) IMU must
	# win over the co-located "Undefined" duplicate. Compiles wmr_config.c directly to test the parser.
	add_executable(tests_wmr_config tests_wmr_config.cpp
									${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/drivers/wmr/wmr_config.c)
	target_link_libraries(
		tests_wmr_config PRIVATE xrt-external-catch2 xrt-interfaces aux_math aux_util xrt-external-cjson)
	target_include_directories(tests_wmr_config PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/drivers)
	add_test(NAME tests_wmr_config COMMAND tests_wmr_config --success)
	set_target_properties(tests_wmr_config PROPERTIES FOLDER monado_tests)
endif()

# Standalone telemetry stress + correctness test (own main, stubs u_log, compiles u_g2_telemetry.c
# directly to stay self-contained — NOT linked against aux_util). A real ctest; guards the wait-free ring
# + the manifest offsets (incl. the head_pose stream) that the offline harness depends on.
add_executable(tests_g2_telemetry tests_g2_telemetry.c
	${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/auxiliary/util/u_g2_telemetry.c)
target_include_directories(
	tests_g2_telemetry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/include
							   ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/auxiliary)
target_link_libraries(tests_g2_telemetry PRIVATE Threads::Threads)
target_compile_definitions(tests_g2_telemetry PRIVATE _GNU_SOURCE) # pthread_setname_np in os_threading.h
add_test(NAME tests_g2_telemetry COMMAND tests_g2_telemetry ${CMAKE_CURRENT_BINARY_DIR}/g2_telem_test_out)
set_target_properties(tests_g2_telemetry PROPERTIES FOLDER monado_tests)

# Standalone C-mode compile + exercise of the wmr_protocol.h byte readers (own main; guards the
# sign-extension contract and keeps the header C11-clean for negative device bytes — audit R4#3).
add_executable(tests_wmr_protocol tests_wmr_protocol.c)
target_include_directories(
	tests_wmr_protocol PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/drivers
							   ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/include
							   ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/auxiliary)
add_test(NAME tests_wmr_protocol COMMAND tests_wmr_protocol)
set_target_properties(tests_wmr_protocol PROPERTIES FOLDER monado_tests)

if(XRT_BUILD_DRIVER_HANDTRACKING)
	target_link_libraries(
		tests_levenbergmarquardt
		PRIVATE
			aux_math
			t_ht_mercury_includes
			t_ht_mercury_kine_lm_includes
			t_ht_mercury
			t_ht_mercury_kine_lm
		)
endif()

if(XRT_HAVE_D3D11)
	target_link_libraries(tests_aux_d3d_d3d11 PRIVATE aux_d3d)
	target_link_libraries(tests_comp_client_d3d11 PRIVATE comp_client comp_mock)
endif()

if(XRT_HAVE_D3D12)
	target_link_libraries(tests_aux_d3d_d3d12 PRIVATE aux_d3d)
	target_link_libraries(tests_comp_client_d3d12 PRIVATE comp_client comp_mock)
endif()

if(XRT_HAVE_VULKAN)
	target_link_libraries(
		tests_comp_client_vulkan PRIVATE comp_client comp_mock comp_util aux_vk
		)
	target_link_libraries(tests_uv_to_tangent PRIVATE comp_render)
endif()

if(XRT_FEATURE_OPENXR)
	target_link_libraries(
		tests_input_transform PRIVATE st_oxr xrt-interfaces xrt-external-openxr
		)
endif()
if(_have_opengl_test)
	target_link_libraries(
		tests_comp_client_opengl PRIVATE comp_client comp_mock aux_ogl SDL2::SDL2
		)
	target_include_directories(tests_comp_client_opengl PRIVATE SDL2::SDL2)
endif()

if(XRT_HAVE_VULKAN AND XRT_HAVE_D3D11)
	target_link_libraries(tests_aux_d3d_d3d11 PRIVATE comp_util aux_vk)
endif()

if(XRT_HAVE_VULKAN AND XRT_HAVE_D3D12)
	target_link_libraries(tests_aux_d3d_d3d12 PRIVATE comp_util aux_vk)
endif()
