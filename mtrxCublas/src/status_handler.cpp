/*
 * Copyright 2021 - 2022 Marcin Matula
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

// clang-format off
#include <mtrxCore/to_string.hpp>
#include <mtrxCublas/to_string.hpp>
#include <mtrxCore/status_handler.hpp>
#include <mtrxCublas/status_handler.hpp>
// clang-format on

namespace mtrx {
void handleStatus(cublasStatus_t status) {
  handleStatus(status, CUBLAS_STATUS_SUCCESS);
}

void handleStatus(CUresult curesult) { handleStatus(curesult, CUDA_SUCCESS); }

void handleStatus(cudaError_t error) { handleStatus(error, cudaSuccess); }
} // namespace mtrx
