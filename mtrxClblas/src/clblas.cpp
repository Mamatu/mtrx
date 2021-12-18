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

#include <cassert>
#include <cstdio>
#include <exception>

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/opencl.h>
#include <clBLAS.h>
#include <mtrxClblas/clblas.hpp>

#include <mtrxCore/size_of.hpp>
#include <type_traits>

namespace mtrx {
struct Mem {
  void *ptr;
  int count;
  ValueType valueType;
};

Clblas::Clblas() {
  cl_int err;

  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};
  cl_context ctx = 0;
  cl_command_queue queue = 0;

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs(1, &platform, NULL);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
  queue = clCreateCommandQueue(ctx, device, 0, &err);

  /* Setup clBLAS */
  err = clblasSetup(); // handleStatus (cublasCreate(&m_handle));
}

Clblas::~Clblas() {
  try {
    // handleStatus(cublasDestroy(m_handle));
  } catch (const std::exception &ex) {
    fprintf(stderr, "%s\n", ex.what());
    abort();
  }
}

Mem *Clblas::_createMem(size_t count, ValueType valueType) {
  if (count <= 0) {
    std::stringstream sstream;
    sstream << "Cannot created mem with count: " << count;
    throw std::runtime_error(sstream.str());
  }

  Mem *mem = new Mem();

  try {
    mem->ptr = nullptr;
    mem->count = count;

    // auto error = cudaMalloc(&mem->ptr, SizeOf<floatt>(rows*columns));
    // handleStatus(error);
  } catch (const std::exception &ex) {
    delete mem;
  }
  return mem;
}

void Clblas::_destroy(const Mem *mem) {
  // cudaFree(mem->ptr);
  delete mem;
}

uintt Clblas::_getCount(const Mem *mem) const { return mem->count; }

void Clblas::_copyHostToKernel(Mem *mem, void *array) {
  // cublasSetVector(mem->length(), SizeOf<floatt>(), array, 1, mem->ptr, 1);
}

void Clblas::_copyKernelToHost(void *array, Mem *mem) {
  // cublasSetVector(mem->length(), SizeOf<floatt>(), array, 1, mem->ptr, 1);
}

uintt Clblas::_amax(const Mem *mem) {
  const uintt n = mem->count;

  int resultIdx = -1;

  using T =
      std::remove_const<std::remove_pointer<decltype(mem->ptr)>::type>::type;
  // call<T>(cublas_amax, m_handle, n, mem->ptr, 1, &resultIdx);

  if (resultIdx == -1) {
    throw std::runtime_error("resultIdx is -1");
  }

  return resultIdx - 1;
}

uintt Clblas::_amin(const Mem *mem) {
  const uintt n = mem->count;

  int resultIdx = -1;

  using T =
      std::remove_const<std::remove_pointer<decltype(mem->ptr)>::type>::type;
  // call<T>(cublas_amin, m_handle, n, mem->ptr, 1, &resultIdx);

  if (resultIdx == -1) {
    throw std::runtime_error("resultIdx is -1");
  }

  return resultIdx - 1;
}

void Clblas::_rot(Mem *x, Mem *y, void *c, ValueType cType, void *s,
                  ValueType sType) {}

void Clblas::_rot(Mem *x, Mem *y, Mem *c, Mem *s) {}

} // namespace mtrx
