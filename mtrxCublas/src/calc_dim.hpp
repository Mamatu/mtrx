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

#ifndef MTRX_DIM_COUNT_HPP
#define MTRX_DIM_COUNT_HPP

#include <array>

#include <cuComplex.h>

namespace mtrx {

enum class CalcDimStrategy { MAXIMIZE_BLOCK_DIM_USAGE };
/*
 * *
 * @brief Calculate dim of m x n matrix
 * @param thread - threads outcome
 * @param blocks - blocks outcome
 * @param m - rows
 * @param n - columns
 * @param blockDim - block Dim supported by device
 * @param gridDim - grid dim supported by device
 * @param maxThreadsPerBlocks - max threads per block supported by device
 * @param calcDimStrategy - strategy for blocks and threads count calculation
 */
void calculateDim(dim3 &threads, dim3 &blocks, int m, int n,
                  const std::array<int, 3> &blockDim,
                  const std::array<int, 3> &gridDim, int maxThreadsPerBlock,
                  CalcDimStrategy calcDimStrategy =
                      CalcDimStrategy::MAXIMIZE_BLOCK_DIM_USAGE);
} // namespace mtrx

#endif
