// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "VoxelHashMap.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <sophus/se3.hpp>
#include <vector>

#include "VoxelUtils.hpp"

namespace {
using kiss_icp::Voxel;

std::vector<Voxel> GetAdjacentVoxels(const Voxel &voxel, int adjacent_voxels = 1) {
    std::vector<Voxel> voxel_neighborhood;
    for (int i = voxel.x() - adjacent_voxels; i < voxel.x() + adjacent_voxels + 1; ++i) {
        for (int j = voxel.y() - adjacent_voxels; j < voxel.y() + adjacent_voxels + 1; ++j) {
            for (int k = voxel.z() - adjacent_voxels; k < voxel.z() + adjacent_voxels + 1; ++k) {
                voxel_neighborhood.emplace_back(i, j, k);
            }
        }
    }
    return voxel_neighborhood;
}
}  // namespace

namespace kiss_icp {

std::tuple<Eigen::Vector4d, double> VoxelHashMap::GetClosestNeighbor(const Eigen::Vector4d &query,
                                                                     double max_distance) const {
    // Convert the point to voxel coordinates
    const auto &voxel = PointToVoxel(query, voxel_size_);
    // Get nearby voxels on the map
    const auto &query_voxels = GetAdjacentVoxels(voxel);

    // Define metric
    bool use_intensity_metric = this->use_intensity_metric_;
    const auto metric = [&use_intensity_metric, &max_distance](const Eigen::Vector4d &lhs,
                                                               const Eigen::Vector4d &rhs) {
        double intensity_diff;
        if (use_intensity_metric) {
            intensity_diff = abs(lhs.w() - rhs.w());
            if (intensity_diff < 1e-3) intensity_diff = 1e-3;
        } else {
            intensity_diff = 1.0;
        }

        double euclidean_dist = (lhs - rhs).head<3>().norm();
        // If we're outside the max distance in euclidean norm, return a very large number
        if (euclidean_dist > max_distance) {
            return std::numeric_limits<double>::max();
        } else {
            return euclidean_dist * intensity_diff;
        }
    };

    // Find the nearest neighbor
    Eigen::Vector4d closest_neighbor = Eigen::Vector4d::Zero();
    double closest_distance = std::numeric_limits<double>::max();
    std::for_each(query_voxels.cbegin(), query_voxels.cend(), [&](const auto &query_voxel) {
        auto search = map_.find(query_voxel);
        if (search != map_.end()) {
            const auto &points = search.value();
            const Eigen::Vector4d &neighbor =
                *std::min_element(points.cbegin(), points.cend(),
                                  [&](const Eigen::Vector4d &lhs, const Eigen::Vector4d &rhs) {
                                      return metric(lhs, query) < metric(rhs, query);
                                  });

            double distance = metric(neighbor, query);
            if (distance < closest_distance) {
                closest_neighbor = neighbor;
                closest_distance = distance;
            }
        }
    });

    return std::make_tuple(closest_neighbor, closest_distance);
}

std::vector<Eigen::Vector4d> VoxelHashMap::Pointcloud() const {
    std::vector<Eigen::Vector4d> points;
    points.reserve(map_.size() * static_cast<size_t>(max_points_per_voxel_));
    std::for_each(map_.cbegin(), map_.cend(), [&](const auto &map_element) {
        const auto &voxel_points = map_element.second;
        points.insert(points.end(), voxel_points.cbegin(), voxel_points.cend());
    });
    points.shrink_to_fit();
    return points;
}

void VoxelHashMap::Update(const std::vector<Eigen::Vector4d> &points,
                          const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const std::vector<Eigen::Vector4d> &points, const Sophus::SE3d &pose) {
    std::vector<Eigen::Vector4d> points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const Eigen::Vector4d &point) {
                       Eigen::Vector4d out = point;
                       out.head<3>() = pose * point.head<3>();
                       return out;
                   });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector4d> &points) {
    const double map_resolution = std::sqrt(voxel_size_ * voxel_size_ / max_points_per_voxel_);
    std::for_each(points.cbegin(), points.cend(), [&](const Eigen::Vector4d &point) {
        const auto voxel = PointToVoxel(point, voxel_size_);
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_points = search.value();
            if (voxel_points.size() == max_points_per_voxel_ ||
                std::any_of(voxel_points.cbegin(), voxel_points.cend(),
                            [&](const Eigen::Vector4d &voxel_point) {
                                return (voxel_point - point).head<3>().norm() < map_resolution;
                            })) {
                return;
            }
            voxel_points.emplace_back(point);
        } else {
            std::vector<Eigen::Vector4d> voxel_points;
            voxel_points.reserve(max_points_per_voxel_);
            voxel_points.emplace_back(point);
            map_.insert({voxel, std::move(voxel_points)});
        }
    });
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const auto max_distance2 = max_distance_ * max_distance_;
    for (auto it = map_.begin(); it != map_.end();) {
        const auto &[voxel, voxel_points] = *it;
        const Eigen::Vector4d &pt = voxel_points.front();
        if ((pt.head<3>() - origin).squaredNorm() >= (max_distance2)) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}
}  // namespace kiss_icp
