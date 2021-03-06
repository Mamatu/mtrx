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

#ifndef MTRX_CUBLAS_TO_STRING_HPP
#define MTRX_CUBLAS_TO_STRING_HPP

// clang-format off
#include <mtrxCore/to_string.hpp>
#include "cuComplex.h"
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <type_traits>
// clang-format on

namespace mtrx {
std::string toString(cudaError_t error);
std::string toString(CUresult curesult, bool noException = false);
std::string toString(cublasStatus_t status);

template <typename T> std::string toString() {
  if (std::is_same<T, float>::value) {
    return "float";
  }
  if (std::is_same<T, double>::value) {
    return "double";
  }
  if (std::is_same<T, cuComplex>::value) {
    return "cuComplex";
  }
  if (std::is_same<T, cuDoubleComplex>::value) {
    return "cuDoubleComplex";
  }
  return "";
}
} // namespace mtrx
#endif
