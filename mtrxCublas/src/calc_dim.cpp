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

namespace {
void calculateDim_maximizeBlockDimUsage(dim3 &threads, dim3 &blocks, int m,
                                        int n,
                                        const std::array<int, 3> &blockDim,
                                        const std::array<int, 3> &gridDim,
                                        int maxThreadsPerBlock) {

  auto calcThreadsBlocksCount = [](auto matrixDim, auto blockDim) {
    if (matrixDim <= blockDim) {
      return std::make_pair(matrixDim, 1);
    }

    auto div = matrixDim / blockDim;
    auto modulo = matrixDim % blockDim;

    if (modulo != 0) {
      div = div + 1;
    }

    return std::make_pair(blockDim, div);
  };

  threads = {1, 1, 1};
  blocks = {1, 1, 1};

  auto y = calcThreadsBlocksCount(m, blockDim[1]);
  auto x = calcThreadsBlocksCount(n, blockDim[0]);

  auto check = [](auto calculatedGridDim, auto gridDim,
                  const std::string &label) {
    if (calculatedGridDim > gridDim) {
      std::stringstream sstream;
      sstream << "Exceeds grid dimension in " << label << " axis (";
      sstream << calculatedGridDim << " > " << gridDim;
      sstream << ")";
      throw std::runtime_error(sstream.str());
    }
  };

  check(x.second, gridDim[0], "x");
  check(y.second, gridDim[1], "y");

  if (x.first * y.first > maxThreadsPerBlock) {
    std::stringstream sstream;
    sstream << "Calculated threads count per block " << x.first * y.first;
    sstream << "is higher than maxThreadsPerBlock " << maxThreadsPerBlock;
    throw std::runtime_error(sstream.str());
  }

  threads.x = x.first;
  blocks.x = x.second;

  threads.y = y.first;
  blocks.y = y.second;
}
} // namespace

namespace mtrx {

void calculateDim(dim3 &threads, dim3 &blocks, int m, int n,
                  const std::array<int, 3> &blockDim,
                  const std::array<int, 3> &gridDim, int maxThreadsPerBlock,
                  CalcDimStrategy calcDimStrategy) {

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

  using Func = std::function<void(dim3 & threads, dim3 & blocks, int m, int n,
                                  const std::array<int, 3> &blockDim,
                                  const std::array<int, 3> &gridDim,
                                  int maxThreadsPerBlock)>;

  auto maximizeBlockDimUsage = [](auto &threads, auto &blocks, auto m, auto n,
                                  const auto &blockDim, const auto &gridDim,
                                  auto maxThreadsPerBlock) {
    calculateDim_maximizeBlockDimUsage(threads, blocks, m, n, blockDim, gridDim,
                                       maxThreadsPerBlock);
  };

  std::map<CalcDimStrategy, Func> strategies;
  strategies.insert({CalcDimStrategy::MAXIMIZE_BLOCK_DIM_USAGE,
                     std::move(maximizeBlockDimUsage)});

  const auto &func = strategies[calcDimStrategy];
  func(threads, blocks, m, n, blockDim, gridDim, maxThreadsPerBlock);
}
} // namespace mtrx
