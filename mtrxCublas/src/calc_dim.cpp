/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of mtrx.
 *
 * mtrx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * mtrx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with mtrx.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "calc_dim.hpp"
#include <functional>
#include <map>
#include <spdlog/spdlog.h>
#include <sstream>
#include <utility>

namespace mtrx {
void calculateDim(dim3 &threads, dim3 &blocks,
                  int m, int n, const std::array<int, 3> &blockDim,
                  const std::array<int, 3> &gridDim, int maxThreadsPerBlock) {

  if (m == 0 || n == 0) {
    std::stringstream sstream;
    sstream << "Invalid dimension of matrix " << m << "x" << n;
    throw std::runtime_error(sstream.str());
  }

  if (blockDim[0] == 0 || blockDim[1] == 0 || blockDim[2] == 0) {
    std::stringstream sstream;
    sstream << "Invalid dimension of block dim ";
    sstream << blockDim[0] << "x" << blockDim[1] << "x" << blockDim[2];
    throw std::runtime_error(sstream.str());
  }

  if (blockDim[0] == 0 || blockDim[1] == 0 || blockDim[2] == 0) {
    std::stringstream sstream;
    sstream << "Invalid dimension of grid dim ";
    sstream << gridDim[0] << "x" << gridDim[1] << "x" << gridDim[2];
    throw std::runtime_error(sstream.str());
  }

  std::map<int, std::tuple<int, int, int, int>, std::less<int>> diffs;

  int m1 = (m * n <= maxThreadsPerBlock) ? m * n : maxThreadsPerBlock;
  int n1 = 1;

  auto calc_count = [](int a, int a1) {
    int count_a = a / a1;
    if (count_a < 1) {
      count_a = 1;
    }
    return count_a;
  };

  while (m1 >= 1) {
    int count_m = calc_count(m, m1);
    int count_n = calc_count(n, n1);

    if (m1 <= blockDim[0] && n1 <= blockDim[1] && count_m <= gridDim[0] &&
        count_n <= gridDim[1] && m1 * n1 <= maxThreadsPerBlock) {
      int s = m * n;
      int s1 = m1 * count_m * n1 * count_n;
      int diff = s1 - s;
      diffs[diff] = std::make_tuple(m1, n1, count_m, count_n);
    }
    m1 = m1 / 2;
    n1 = n1 * 2;
  }

  spdlog::debug(
      "{} m {} n {} blockDim {} {} gridDim {} {} maxThreadsPerBlock {}",
      __func__, m, n, blockDim[0], blockDim[1], gridDim[0], gridDim[1],
      maxThreadsPerBlock);

  auto front = diffs.begin();
  if (front == diffs.end()) {
    std::stringstream sstream;
    sstream << __func__ << "blockDim " << blockDim[0] << ", " << blockDim[1];
    sstream << " gridDim " << gridDim[0] << ", " << gridDim[1];
    sstream << " maxThreadsPerBlock " << maxThreadsPerBlock;
    sstream << " - not found any calculated dim.";
    spdlog::error(sstream.str());
    throw std::runtime_error(sstream.str());
  }

  threads.y = std::get<0>(front->second);
  threads.x = std::get<1>(front->second);
  threads.z = 1;
  blocks.y = std::get<3>(front->second);
  blocks.x = std::get<2>(front->second);
  blocks.z = 1;

  spdlog::info("{} threads {} {} blocks {} {} blockDim {} {} gridDim {} {} "
               "maxThreadsPerBlock {}",
               __func__, threads.x, threads.y, blocks.x, blocks.y,
               blockDim[0], blockDim[1], gridDim[0], gridDim[1],
               maxThreadsPerBlock);
}
} // namespace mtrx
