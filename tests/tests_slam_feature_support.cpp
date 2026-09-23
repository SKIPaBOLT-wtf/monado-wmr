// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
#include "catch_amalgamated.hpp"

#include "tracking/t_slam_feature_support.hpp"

#include <array>
#include <limits>

TEST_CASE("equal feature counts can have very different two-axis support")
{
	std::array<vit_pose_feature_t, 4> spread{{{1, 0, 0, 1}, {2, 100, 0, 1},
	                                          {3, 0, 100, 1}, {4, 100, 100, 1}}};
	std::array<vit_pose_feature_t, 4> clustered{{{1, 49, 49, 1}, {2, 50, 49, 1},
	                                             {3, 49, 50, 1}, {4, 50, 50, 1}}};
	const auto wide = t_slam_summarize_feature_support({4, spread.data()});
	const auto narrow = t_slam_summarize_feature_support({4, clustered.data()});
	REQUIRE(wide.available);
	REQUIRE(narrow.available);
	REQUIRE(wide.count == narrow.count);
	CHECK(wide.major_rms == Catch::Approx(50));
	CHECK(wide.minor_rms == Catch::Approx(50));
	CHECK(narrow.major_rms == Catch::Approx(0.5));
	CHECK(narrow.minor_rms == Catch::Approx(0.5));
}

TEST_CASE("line, missing, and nonfinite support remain distinguishable")
{
	std::array<vit_pose_feature_t, 3> line{{{1, 0, 0, 1}, {2, 50, 0, 1}, {3, 100, 0, 1}}};
	const auto one_axis = t_slam_summarize_feature_support({3, line.data()});
	CHECK(one_axis.major_rms > 0);
	CHECK(one_axis.minor_rms == Catch::Approx(0));

	line[2].u = std::numeric_limits<float>::quiet_NaN();
	const auto nonfinite = t_slam_summarize_feature_support({3, line.data()});
	CHECK(nonfinite.count == 3);
	CHECK(nonfinite.finite_count == 2);

	const auto zero = t_slam_summarize_feature_support({0, nullptr});
	CHECK(zero.available);
	CHECK(zero.count == 0);
	const auto missing = t_slam_summarize_feature_support({3, nullptr});
	CHECK_FALSE(missing.available);
	CHECK(missing.count == -1);
}
