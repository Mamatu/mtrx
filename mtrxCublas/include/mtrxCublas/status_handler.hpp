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

#ifndef MTRX_CUBLAS_STATUS_HANDLER_HPP
#define MTRX_CUBLAS_STATUS_HANDLER_HPP

// clang-format off
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
// clang-format on

namespace mtrx {
void handleStatus(cublasStatus_t status);
void handleStatus(CUresult curesult, bool noException = false);
void handleStatus(cudaError_t error);
} // namespace mtrx

#endif
