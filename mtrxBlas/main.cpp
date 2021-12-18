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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mtrxCore/size_of.hpp>
#include <mtrxCublas/status_handler.hpp>

int main() {
  using namespace mtrx;

  cublasHandle_t handle;
  auto status = cublasCreate(&handle);
  handleStatus(status);

  float *d_A = nullptr;
  int n2 = 3;

  float *h_A = new float[n2];
  for (int x = 0; x < n2; ++x) {
    h_A[x] = x;
  }

  auto error =
      cudaMalloc(reinterpret_cast<void **>(&d_A), SizeOf<decltype(d_A)>(n2));
  handleStatus(error);

  status = cublasSetVector(n2, SizeOf<decltype(h_A)>(), h_A, 1, d_A, 1);
  handleStatus(status);

  int result = -1;
  status = cublasIsamax(handle, n2, d_A, 1, &result);
  handleStatus(status);

  printf("result %d \n", result);

  status = cublasDestroy(handle);
  handleStatus(status);

  return 0;
}
