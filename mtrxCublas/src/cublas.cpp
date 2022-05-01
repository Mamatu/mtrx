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

#include "cuComplex.h"
#include "driver_types.h"
#include "mtrxCore/status_handler.hpp"
#include "mtrxCore/types.hpp"
#include <cassert>
#include <cstdio>
#include <exception>

#include <cuda_runtime.h>
#include <mtrxCore/checkers.hpp>
#include <mtrxCublas/cublas.hpp>

#include <map>
#include <mtrxCore/size_of.hpp>
#include <mtrxCublas/status_handler.hpp>
#include <mtrxCublas/to_string.hpp>
#include <spdlog/fmt/bundled/format-inl.h>
#include <spdlog/spdlog.h>
#include <type_traits>

#include "cuda_alloc.hpp"
#include "kernels.hpp"

#ifdef CUBLAS_NVPROF_KERNELS
#include "cuda_profiler.hpp"
#define PROFILER() Profiler p;
#else
#define PROFILER()
#endif

namespace mtrx {
struct Mem {
  void *ptr = nullptr;
  int count = 0;
  ValueType valueType = ValueType::NOT_DEFINED;
};

template <typename T> void *cublas_getOffset(Mem *mem, int idx) {
  T *ptr = static_cast<T *>(mem->ptr);
  return static_cast<void *>(&ptr[idx]);
}

void *cublas_getOffset(Mem *mem, int idx) {
  if (mem->valueType == ValueType::FLOAT) {
    return cublas_getOffset<float>(mem, idx);
  } else if (mem->valueType == ValueType::DOUBLE) {
    return cublas_getOffset<double>(mem, idx);
  } else if (mem->valueType == ValueType::FLOAT_COMPLEX) {
    return cublas_getOffset<cuComplex>(mem, idx);
  } else if (mem->valueType == ValueType::DOUBLE_COMPLEX) {
    return cublas_getOffset<cuDoubleComplex>(mem, idx);
  }
  return nullptr;
}

int SizeOf(ValueType valueType, int n = 1) {
  switch (valueType) {
  case ValueType::FLOAT:
    return sizeof(float) * n;
  case ValueType::DOUBLE:
    return sizeof(double) * n;
  case ValueType::FLOAT_COMPLEX:
    return sizeof(cuComplex) * n;
  case ValueType::DOUBLE_COMPLEX:
    return sizeof(cuDoubleComplex) * n;
  case ValueType::NOT_DEFINED:
    throw std::runtime_error("Not defined value type");
  };
  return 0;
}

int SizeOfPtr(ValueType valueType, int n = 1) {
  switch (valueType) {
  case ValueType::FLOAT:
    return sizeof(float *) * n;
  case ValueType::DOUBLE:
    return sizeof(double *) * n;
  case ValueType::FLOAT_COMPLEX:
    return sizeof(cuComplex *) * n;
  case ValueType::DOUBLE_COMPLEX:
    return sizeof(cuDoubleComplex *) * n;
  case ValueType::NOT_DEFINED:
    throw std::runtime_error("Not defined value type");
  };
  return 0;
}

cublasOperation_t convert(Operation operation) {
  switch (operation) {
  case Operation::OP_N:
    return CUBLAS_OP_N;
  case Operation::OP_T:
    return CUBLAS_OP_T;
  case Operation::OP_C:
    return CUBLAS_OP_C;
  };

  throw std::runtime_error("Not defined side mode");
  return CUBLAS_OP_N;
}

cublasSideMode_t convert(SideMode sideMode) {
  switch (sideMode) {
  case SideMode::LEFT:
    return CUBLAS_SIDE_LEFT;
  case SideMode::RIGHT:
    return CUBLAS_SIDE_RIGHT;
  };

  throw std::runtime_error("Not defined side mode");
  return CUBLAS_SIDE_LEFT;
}

cublasFillMode_t convert(FillMode fillMode) {
  switch (fillMode) {
  case FillMode::FULL:
    return CUBLAS_FILL_MODE_FULL;
  case FillMode::LOWER:
    return CUBLAS_FILL_MODE_LOWER;
  case FillMode::UPPER:
    return CUBLAS_FILL_MODE_UPPER;
  }

  throw std::runtime_error("Not defined fill mode");
  return CUBLAS_FILL_MODE_UPPER;
}

template <typename T>
T *cublas_allocCudaArrayCopy(const T *array, size_t count) {
  T *d_array = nullptr;
  auto error = cudaMalloc(&d_array, sizeof(T) * count);
  handleStatus(error);

  auto error1 =
      cudaMemcpy(d_array, array, sizeof(T) * count, cudaMemcpyHostToDevice);
  handleStatus(error1);

  return d_array;
}

template <typename T> T *cublas_allocCudaArrayCopy(const std::vector<T> &vec) {
  return cublas_allocCudaArrayCopy(vec.data(), vec.size());
}

template <typename T> void cublas_deallocCudaArray(T *array) {
  auto error = cudaFree(array);
  handleStatus(error);
}

Cublas::Cublas() { handleStatus(cublasCreate(&m_handle)); }

Cublas::~Cublas() {
  try {
    handleStatus(cublasDestroy(m_handle));
  } catch (const std::exception &ex) {
    spdlog::error("%s", ex.what());
    abort();
  }
}

std::vector<int> Cublas::_getDevices() const {
  int count = 0;
  handleStatus(cudaGetDeviceCount(&count));

  std::vector<int> devices;
  devices.reserve(count);

  for (int i = 0; i < count; ++i) {
    devices.push_back(i);
  }

  return devices;
}

void Cublas::_setDevice(int device) { handleStatus(cudaSetDevice(device)); }

template <typename T, typename Vec>
void cublas_setVec(Vec &&vec, size_t rows, size_t columns) {
  vec.reserve(rows * columns);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < columns; ++c) {
      if (r == c) {
        vec.push_back(static_cast<T>(1));
      } else {
        vec.push_back(static_cast<T>(0));
      }
    }
  }
}

template <typename T>
Mem *cublas_createIdentity(Cublas *cublas, size_t rows, size_t columns,
                           ValueType valueType) {
  Mem *matrix = cublas->createMatrix(rows, columns, valueType);
  std::vector<T> vec;
  cublas_setVec<T>(vec, rows, columns);
  cublas->copyHostToKernel(matrix, vec.data());
  return matrix;
}

Mem *Cublas::_createIdentityMatrix(size_t rows, size_t columns,
                                   ValueType valueType) {
  if (valueType == ValueType::FLOAT) {
    return cublas_createIdentity<float>(this, rows, columns, valueType);
  } else if (valueType == ValueType::DOUBLE) {
    return cublas_createIdentity<double>(this, rows, columns, valueType);
  }

  throw std::runtime_error("Not supported type");
  return nullptr;
}

Mem *Cublas::_createMem(size_t count, ValueType valueType) {
  if (count <= 0) {
    std::stringstream sstream;
    sstream << "Cannot created mem with count: " << count;
    throw std::runtime_error(sstream.str());
  }

  Mem *mem = new Mem();

  try {
    mem->ptr = nullptr;
    mem->count = count;
    mem->valueType = valueType;

    auto error = cudaMalloc(&(mem->ptr), SizeOf(valueType, mem->count));
    handleStatus(error);
    auto error1 = cudaMemset(mem->ptr, 0, SizeOf(valueType, mem->count));
    handleStatus(error1);
  } catch (const std::exception &ex) {
    delete mem;
    throw std::runtime_error(ex.what());
  }
  return mem;
}

void Cublas::_destroy(const Mem *mem) {
  auto error = cudaFree(mem->ptr);
  handleStatus(error);
  delete mem;
}

uintt Cublas::_getCount(const Mem *mem) const { return mem->count; }

uintt Cublas::_getSizeInBytes(const Mem *mem) const {
  return SizeOf(mem->valueType, mem->count);
}

void Cublas::_copyHostToKernel(Mem *mem, void *array) {
  auto status = cublasSetVector(mem->count, SizeOf(mem->valueType), array, 1,
                                mem->ptr, 1);
  handleStatus(status);
}

void Cublas::_copyKernelToHost(void *array, Mem *mem) {
  auto status = cublasGetVector(mem->count, SizeOf(mem->valueType), mem->ptr, 1,
                                array, 1);
  handleStatus(status);
}

uintt Cublas::_amax(const Mem *mem) {
  const uintt n = mem->count;

  int resultIdx = -1;

  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
  switch (mem->valueType) {
  case ValueType::FLOAT:
    status = cublasIsamax(m_handle, n, static_cast<float *>(mem->ptr), 1,
                          &resultIdx);
    break;
  case ValueType::DOUBLE:
    status = cublasIdamax(m_handle, n, static_cast<double *>(mem->ptr), 1,
                          &resultIdx);
    break;
  case ValueType::FLOAT_COMPLEX:
    status = cublasIcamax(m_handle, n, static_cast<cuComplex *>(mem->ptr), 1,
                          &resultIdx);
    break;
  case ValueType::DOUBLE_COMPLEX:
    status = cublasIzamax(m_handle, n, static_cast<cuDoubleComplex *>(mem->ptr),
                          1, &resultIdx);
    break;
  case ValueType::NOT_DEFINED:
    throw std::runtime_error("Not defined value type");
  };

  handleStatus(status);

  if (resultIdx == -1) {
    throw std::runtime_error("resultIdx is -1");
  }

  return resultIdx - 1;
}

uintt Cublas::_amin(const Mem *mem) {
  const uintt n = mem->count;

  int resultIdx = -1;

  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
  switch (mem->valueType) {
  case ValueType::FLOAT:
    status = cublasIsamin(m_handle, n, static_cast<float *>(mem->ptr), 1,
                          &resultIdx);
    break;
  case ValueType::DOUBLE:
    status = cublasIdamin(m_handle, n, static_cast<double *>(mem->ptr), 1,
                          &resultIdx);
    break;
  case ValueType::FLOAT_COMPLEX:
    status = cublasIcamin(m_handle, n, static_cast<cuComplex *>(mem->ptr), 1,
                          &resultIdx);
    break;
  case ValueType::DOUBLE_COMPLEX:
    status = cublasIzamin(m_handle, n, static_cast<cuDoubleComplex *>(mem->ptr),
                          1, &resultIdx);
    break;
  case ValueType::NOT_DEFINED:
    throw std::runtime_error("Not defined value type");
  };

  handleStatus(status);

  if (resultIdx == -1) {
    throw std::runtime_error("resultIdx is -1");
  }

  return resultIdx - 1;
}

void Cublas::_rot(Mem *x, Mem *y, void *c, ValueType cType, void *s,
                  ValueType sType) {
  if (cType == ValueType::DOUBLE_COMPLEX || cType == ValueType::FLOAT_COMPLEX) {
    throw std::runtime_error(
        "Not supported cType. Cannot be DOUBLE_MAX or FLOAT_COMPLEX");
  }

  if (x->valueType != y->valueType) {
    std::stringstream sstream;
    sstream << "Different type for x and y. x: " << toString(x->valueType)
            << " y: " << toString(y->valueType);
    throw std::runtime_error(sstream.str());
  }

  if (x->count != y->count) {
    std::stringstream sstream;
    sstream << "Different count for x and y. x: " << x->count
            << " y: " << y->count;
    throw std::runtime_error(sstream.str());
  }

  const auto n = x->count;

  auto _cublasSrot = [this, n](Mem *x, Mem *y, void *c, void *s) {
    return cublasSrot(m_handle, n, static_cast<float *>(x->ptr), 1,
                      static_cast<float *>(y->ptr), 1, static_cast<float *>(c),
                      static_cast<float *>(s));
  };

  auto _cublasDrot = [this, n](Mem *x, Mem *y, void *c, void *s) {
    return cublasDrot(m_handle, n, static_cast<double *>(x->ptr), 1,
                      static_cast<double *>(y->ptr), 1,
                      static_cast<double *>(c), static_cast<double *>(s));
  };

  using Impl = std::function<cublasStatus_t(Mem *, Mem *, void *, void *)>;
  const std::map<std::tuple<ValueType, ValueType, ValueType, ValueType>, Impl>
      impls = {{std::make_tuple(ValueType::FLOAT, ValueType::FLOAT,
                                ValueType::FLOAT, ValueType::FLOAT),
                std::move(_cublasSrot)},
               {std::make_tuple(ValueType::DOUBLE, ValueType::DOUBLE,
                                ValueType::DOUBLE, ValueType::DOUBLE),
                std::move(_cublasDrot)}};

  const auto it =
      impls.find(std::make_tuple(x->valueType, y->valueType, cType, sType));

  if (it == impls.end()) {
    std::stringstream sstream;
    sstream << "Not supported overload: (";
    sstream << toString(x->valueType) << ", ";
    sstream << toString(y->valueType) << ", ";
    sstream << toString(cType) << ", ";
    sstream << toString(sType) << ")";
    throw std::runtime_error(sstream.str());
  }

  auto status = it->second(x, y, c, s);
  handleStatus(status);
}

void Cublas::_rot(Mem *x, Mem *y, Mem *c, Mem *s) {
  _rot(x, y, c->ptr, c->valueType, s->ptr, s->valueType);
}

void Cublas::_syr(FillMode fillMode, Mem *output, void *alpha,
                  ValueType alphaType, Mem *x) {
  int n = getRows(output);
  int m = getColumns(output);
  const int lda = n;

  if (n != m) {
    std::stringstream sstream;
    sstream << "Matrix 'output' must be square not " << n << "x" << m;
    throw std::runtime_error(sstream.str());
  }

  auto _cublasSsyr = [this, n, lda](FillMode fillMode, Mem *output, void *alpha,
                                    Mem *x) {
    auto *output_f = static_cast<float *>(output->ptr);
    auto *alpha_f = static_cast<float *>(alpha);
    auto *x_f = static_cast<float *>(x->ptr);

    return cublasSsyr(m_handle, convert(fillMode), n, alpha_f, x_f, 1, output_f,
                      lda);
  };

  auto _cublasDsyr = [this, n, lda](FillMode fillMode, Mem *output, void *alpha,
                                    Mem *x) {
    auto *output_f = static_cast<double *>(output->ptr);
    auto *alpha_f = static_cast<double *>(alpha);
    auto *x_f = static_cast<double *>(x->ptr);

    return cublasDsyr(m_handle, convert(fillMode), n, alpha_f, x_f, 1, output_f,
                      lda);
  };

  using Impl = std::function<cublasStatus_t(FillMode, Mem *, void *, Mem *)>;
  const std::map<std::tuple<ValueType, ValueType, ValueType>, Impl> impls = {
      {std::make_tuple(ValueType::FLOAT, ValueType::FLOAT, ValueType::FLOAT),
       std::move(_cublasSsyr)},
      {std::make_tuple(ValueType::DOUBLE, ValueType::DOUBLE, ValueType::DOUBLE),
       std::move(_cublasDsyr)}};

  const auto it =
      impls.find(std::make_tuple(output->valueType, alphaType, x->valueType));

  if (it == impls.end()) {
    std::stringstream sstream;
    sstream << "Not supported overload: (";
    sstream << toString(output->valueType) << ", ";
    sstream << toString(alphaType) << ", ";
    sstream << toString(x->valueType) << ")";
    throw std::runtime_error(sstream.str());
  }

  auto call = [it, output, alpha, x](FillMode fillMode) {
    auto status = it->second(fillMode, output, alpha, x);
    handleStatus(status);
  };

  if (fillMode != FillMode::FULL) {
    call(fillMode);
  } else {
    call(FillMode::LOWER);
    call(FillMode::UPPER);
    if (alphaType == ValueType::FLOAT) {
      float scaleFactor = 0.5f;
      scaleDiagonal(output, static_cast<void *>(&scaleFactor),
                    ValueType::FLOAT);
    } else if (alphaType == ValueType::DOUBLE) {
      double scaleFactor = 0.5;
      scaleDiagonal(output, static_cast<void *>(&scaleFactor),
                    ValueType::DOUBLE);
    } else {
      throw std::runtime_error("Not supported yet...");
    }
  }
}

void Cublas::_syr(FillMode fillMode, Mem *output, Mem *alpha, Mem *x) {
  _syr(fillMode, output, alpha->ptr, alpha->valueType, x);
}

void Cublas::_gemm(Mem *output, void *alpha, ValueType alphaType,
                   Operation transa, Mem *a, Operation transb, Mem *b,
                   void *beta, ValueType betaType) {
  const auto m = getRows(a);
  const auto m1 = getRows(output);

  if (m != m1) {
    std::stringstream sstream;
    sstream
        << "Rows of 'a' matrix is not equal to rows of 'output'. Rows('a'): "
        << m << " Rows('output'): " << m1;
    throw std::runtime_error(sstream.str());
  }

  const auto n = getColumns(b);
  const auto n1 = getColumns(output);

  if (n != n1) {
    std::stringstream sstream;
    sstream << "Columns of 'b' matrix is not equal to columns of 'output'. "
               "Columns('b'): "
            << n << " Columns('output'): " << n1;
    throw std::runtime_error(sstream.str());
  }

  const auto k = getColumns(a);
  const auto k1 = getRows(b);

  if (k != k1) {
    std::stringstream sstream;
    sstream << "Columns of 'a' matrix is not equal to rows of 'b' matrix. "
               "Columns('a'): "
            << k << " Rows('b'): " << k1;
    throw std::runtime_error(sstream.str());
  }

  auto cutransa = convert(transa);
  auto cutransb = convert(transb);

  auto _cublasSgemm = [this, m, n, k, cutransa, cutransb](Mem *output,
                                                          void *alpha, Mem *a,
                                                          Mem *b, void *beta) {
    auto *alpha_f = static_cast<float *>(alpha);
    auto *output_f = static_cast<float *>(output->ptr);
    auto *a_f = static_cast<float *>(a->ptr);
    auto *b_f = static_cast<float *>(b->ptr);
    auto *beta_f = static_cast<float *>(beta);
    return cublasSgemm(m_handle, cutransa, cutransb, m, n, k, alpha_f, a_f, m,
                       b_f, k, beta_f, output_f, m);
  };

  auto _cublasDgemm = [this, m, n, k, cutransa, cutransb](Mem *output,
                                                          void *alpha, Mem *a,
                                                          Mem *b, void *beta) {
    auto *alpha_f = static_cast<double *>(alpha);
    auto *output_f = static_cast<double *>(output->ptr);
    auto *a_f = static_cast<double *>(a->ptr);
    auto *b_f = static_cast<double *>(b->ptr);
    auto *beta_f = static_cast<double *>(beta);
    return cublasDgemm(m_handle, cutransa, cutransb, m, n, k, alpha_f, a_f, m,
                       b_f, k, beta_f, output_f, m);
  };

  using Impl =
      std::function<cublasStatus_t(Mem *, void *, Mem *, Mem *, void *)>;
  const std::map<
      std::tuple<ValueType, ValueType, ValueType, ValueType, ValueType>, Impl>
      impls = {
          {std::make_tuple(ValueType::FLOAT, ValueType::FLOAT, ValueType::FLOAT,
                           ValueType::FLOAT, ValueType::FLOAT),
           std::move(_cublasSgemm)},
          {std::make_tuple(ValueType::DOUBLE, ValueType::DOUBLE,
                           ValueType::DOUBLE, ValueType::DOUBLE,
                           ValueType::DOUBLE),
           std::move(_cublasDgemm)}};

  const auto it = impls.find(std::make_tuple(
      output->valueType, alphaType, a->valueType, b->valueType, betaType));

  if (it == impls.end()) {
    std::stringstream sstream;
    sstream << "Not supported overload: (";
    sstream << toString(output->valueType) << ", ";
    sstream << toString(alphaType) << ", ";
    sstream << toString(a->valueType) << ", ";
    sstream << toString(b->valueType) << ", ";
    sstream << toString(betaType) << ")";
    throw std::runtime_error(sstream.str());
  }

  auto status = it->second(output, alpha, a, b, beta);
  handleStatus(status);
}

void Cublas::_gemm(Mem *output, Mem *alpha, Operation transa, Mem *a,
                   Operation transb, Mem *b, Mem *beta) {
  _gemm(output, alpha->ptr, alpha->valueType, transa, a, transb, b, beta->ptr,
        beta->valueType);
}

void Cublas::_matrixMul(Mem *output, Mem *a, Mem *b) {
  double alphaD = 1.;
  float alphaF = 1.f;

  double betaD = 0.;
  float betaF = 0.f;

  void *alpha = &alphaF;
  void *beta = &betaF;
  ValueType type = ValueType::FLOAT;

  if (output->valueType == ValueType::DOUBLE) {
    alpha = &alphaD;
    beta = &betaD;
    type = ValueType::DOUBLE;
  }

  _gemm(output, alpha, type, Operation::OP_N, a, Operation::OP_N, b, beta,
        type);
}

void Cublas::_symm(SideMode sideMode, FillMode fillMode, Mem *output,
                   void *alpha, ValueType alphaType, Mem *a, Mem *b, void *beta,
                   ValueType betaType) {
  int m = getRows(output);
  int n = getColumns(output);
  int lda = m; // std::max(1, m);
  int ldb = m; // std::max(1, m);
  int ldc = m;

  auto _cublasSsymm = [this, sideMode, fillMode, m, n, lda, ldb,
                       ldc](Mem *output, void *alpha, Mem *a, Mem *b,
                            void *beta) {
    auto *output_f = static_cast<float *>(output->ptr);
    auto *alpha_f = static_cast<float *>(alpha);
    auto *a_f = static_cast<float *>(a->ptr);
    auto *b_f = static_cast<float *>(b->ptr);
    auto *beta_f = static_cast<float *>(beta);
    return cublasSsymm(m_handle, convert(sideMode), convert(fillMode), m, n,
                       alpha_f, a_f, lda, b_f, ldb, beta_f, output_f, ldc);
  };

  auto _cublasDsymm = [this, sideMode, fillMode, m, n, lda, ldb,
                       ldc](Mem *output, void *alpha, Mem *a, Mem *b,
                            void *beta) {
    auto *output_f = static_cast<double *>(output->ptr);
    auto *alpha_f = static_cast<double *>(alpha);
    auto *a_f = static_cast<double *>(a->ptr);
    auto *b_f = static_cast<double *>(b->ptr);
    auto *beta_f = static_cast<double *>(beta);
    return cublasDsymm(m_handle, convert(sideMode), convert(fillMode), m, n,
                       alpha_f, a_f, lda, b_f, ldb, beta_f, output_f, ldc);
  };

  using Impl =
      std::function<cublasStatus_t(Mem *, void *, Mem *, Mem *, void *)>;
  const std::map<
      std::tuple<ValueType, ValueType, ValueType, ValueType, ValueType>, Impl>
      impls = {
          {std::make_tuple(ValueType::FLOAT, ValueType::FLOAT, ValueType::FLOAT,
                           ValueType::FLOAT, ValueType::FLOAT),
           std::move(_cublasSsymm)},
          {std::make_tuple(ValueType::DOUBLE, ValueType::DOUBLE,
                           ValueType::DOUBLE, ValueType::DOUBLE,
                           ValueType::DOUBLE),
           std::move(_cublasDsymm)}};

  const auto it = impls.find(std::make_tuple(
      output->valueType, alphaType, a->valueType, b->valueType, betaType));

  if (it == impls.end()) {
    std::stringstream sstream;
    sstream << "Not supported overload: (";
    sstream << toString(output->valueType) << ", ";
    sstream << toString(alphaType) << ", ";
    sstream << toString(a->valueType) << ", ";
    sstream << toString(b->valueType) << ", ";
    sstream << toString(betaType) << ")";
    throw std::runtime_error(sstream.str());
  }

  auto status = it->second(output, alpha, a, b, beta);
  handleStatus(status);
}

void Cublas::_symm(SideMode sideMode, FillMode fillMode, Mem *output,
                   Mem *alpha, Mem *a, Mem *b, Mem *beta) {
  _symm(sideMode, fillMode, output, alpha->ptr, alpha->valueType, a, b,
        beta->ptr, beta->valueType);
}

void Cublas::_geqrf(Mem *a, Mem *tau) {
  auto as = Mems{a};
  auto taus = Mems{tau};
  _geqrf(as, taus);
}

void Cublas::_geqrf(Mems &a, Mems &tau) {
  if (a.size() != tau.size()) {
    std::stringstream sstream;
    sstream << "Invalid count of element between 'a' and 'tau'." << a.size()
            << " != " << tau.size();
    throw std::runtime_error(sstream.str());
  }

  if (a.size() == 0) {
    throw std::runtime_error("Array of martices 'a' is empty");
  }

  if (tau.size() == 0) {
    throw std::runtime_error("Array of martices 'tau' is empty");
  }

  auto aValueType = a[0]->valueType;
  auto tauValueType = tau[0]->valueType;

  if (a[0]->valueType != tau[0]->valueType) {
    std::stringstream sstream;
    sstream << "Value type of 'a' is " << toString(a[0]->valueType);
    sstream << " and value type of 'tau' is " << toString(tau[0]->valueType);
    throw std::runtime_error(sstream.str());
  }

  std::vector<void *> _a;
  std::vector<void *> _tau;

  auto checkDim = [](int _rows, int _columns, int rows, int columns) {
    if (_rows != -1 && (rows != _rows || columns != _columns)) {
      std::stringstream sstream;
      sstream << "Invalid dim. ";
      sstream << "Expected: ";
      sstream << "(" << _rows << ", " << _columns << ") ";
      sstream << "Actual: ";
      sstream << "(" << rows << ", " << columns << ")";
      throw std::runtime_error(sstream.str());
    }
  };

  auto checkValueType = [](ValueType _valueType, ValueType valueType) {
    if (_valueType != ValueType::NOT_DEFINED && _valueType != valueType) {
      std::stringstream sstream;
      sstream << "Invalid value type. ";
      sstream << "Expected: " << toString(_valueType) << " ";
      sstream << "Actual: " << toString(valueType);
      throw std::runtime_error(sstream.str());
    }
  };

  auto create = [this, &checkDim, &checkValueType](auto &array,
                                                   const auto &input) {
    int m = -1;
    int n = -1;

    array.reserve(input.size());

    int _rows = -1;
    int _columns = -1;
    ValueType _valueType = ValueType::NOT_DEFINED;

    for (const auto *elem : input) {
      array.push_back(elem->ptr);

      auto rows = getRows(elem);
      auto columns = getColumns(elem);
      auto valueType = elem->valueType;

      checkDim(_rows, _columns, rows, columns);
      checkValueType(_valueType, valueType);

      _rows = rows;
      _columns = columns;
      _valueType = valueType;
    }
    m = _rows;
    n = _columns;
    return std::make_pair(m, n);
  };

  auto pair = create(_a, a);
  create(_tau, tau);

  int m = pair.first;
  int n = pair.second;

  const int lda = m;
  int batchSize = a.size();

  auto _cublasSgeqrfBatched = [this, m, n, lda,
                               batchSize](std::vector<void *> &a,
                                          std::vector<void *> &tau) {
    int info = 0;

    static_assert(sizeof(float *) == sizeof(void *),
                  "Mismatch beetwen size of float* and void*");

    auto **a_f = reinterpret_cast<float **>(a.data());
    auto **tau_f = reinterpret_cast<float **>(tau.data());

    auto **d_a = cublas_allocCudaArrayCopy<float *>(a_f, a.size());
    auto **d_tau = cublas_allocCudaArrayCopy<float *>(tau_f, tau.size());

    auto status =
        cublasSgeqrfBatched(m_handle, m, n, d_a, lda, d_tau, &info, batchSize);

    cublas_deallocCudaArray(d_a);
    cublas_deallocCudaArray(d_tau);

    return std::make_pair(status, info);
  };

  auto _cublasDgeqrfBatched = [this, m, n, lda,
                               batchSize](std::vector<void *> &a,
                                          std::vector<void *> &tau) {
    int info = 0;
    double **a_f = reinterpret_cast<double **>(a.data());
    double **tau_f = reinterpret_cast<double **>(tau.data());

    auto **d_a = cublas_allocCudaArrayCopy<double *>(a_f, a.size());
    auto **d_tau = cublas_allocCudaArrayCopy<double *>(tau_f, tau.size());

    auto status =
        cublasDgeqrfBatched(m_handle, m, n, d_a, lda, d_tau, &info, batchSize);

    cublas_deallocCudaArray(d_a);
    cublas_deallocCudaArray(d_tau);

    return std::make_pair(status, info);
  };

  using Impl = std::function<std::pair<cublasStatus_t, int>(
      std::vector<void *> &, std::vector<void *> &)>;
  const std::map<std::tuple<ValueType, ValueType>, Impl> impls = {
      {std::make_tuple(ValueType::FLOAT, ValueType::FLOAT),
       std::move(_cublasSgeqrfBatched)},
      {std::make_tuple(ValueType::DOUBLE, ValueType::DOUBLE),
       std::move(_cublasDgeqrfBatched)}};

  const auto it = impls.find(std::make_tuple(aValueType, tauValueType));

  if (it == impls.end()) {
    std::stringstream sstream;
    sstream << "Not supported overload: (";
    sstream << toString(aValueType) << ", ";
    sstream << toString(tauValueType) << ")";
    throw std::runtime_error(sstream.str());
  }

  auto pairStatus = it->second(_a, _tau);

  handleStatus(pairStatus.first);

  if (pairStatus.second < 0) {
    std::stringstream sstream;
    sstream << "The parameters passed at " << -pairStatus.second
            << " is invalid";
    throw std::runtime_error(sstream.str());
  }
}

template <typename T>
void cublas_qrDecomposition(Cublas *cublas, Blas::Mems &q, Blas::Mems &r,
                            Blas::Mems &a, ValueType valueType) {
  auto m = cublas->getRows(a[0]);
  auto n = cublas->getColumns(a[0]);
  auto k = std::min(m, n);

  if (m != n) {
    std::stringstream sstream;
    sstream << "Matrix is not square! It is " << m << "x" << n;
    throw std::runtime_error(sstream.str());
  }

  auto m2 = m * n;
  std::vector<T> h_zeros(m2, static_cast<T>(0));

  Mem *I = cublas->createIdentityMatrix(m, m, valueType);
  Mem *H = cublas->createMatrix(m, m, valueType);
  Mem *Aux1 = cublas->createMatrix(m, m, valueType);
  Mem *Aux2 = cublas->createMatrix(m, m, valueType);
  Mem *v = cublas->create(m, valueType);

  Blas::Mems aux3;
  for (auto *mema : a) {
    Mem *Aux3 = cublas->createMatrix(m, m, valueType);
    cublas->matrixMul(Aux3, I, mema);
    aux3.push_back(Aux3);
  }

  auto dim = std::max(static_cast<size_t>(1), std::min(m, n));

  Blas::Mems taus;
  taus.reserve(a.size());
  for (size_t idx = 0; idx < a.size(); ++idx) {
    taus.push_back(cublas->create(dim, a[0]->valueType));
  }

  cublas->geqrf(aux3, taus);
  std::vector<T> h_taus;
  h_taus.resize(dim);

  for (size_t j = 0; j < aux3.size(); ++j) {
    cublas->copyKernelToHost(h_taus.data(), taus[j]);
    for (size_t i = 0; i < k; ++i) {

      auto *d_array = aux3[j];
      auto rows = cublas->getRows(d_array);

      int idx1 = (i + 1) + rows * i;
      int idx2 = (m) + rows * i;
      int offset = idx2 - idx1;
      Mem mem = {cublas_getOffset(d_array, idx1), offset, valueType};

      std::vector<T> h_v(m, 0);
      h_v[i] = 1;
      auto status = cudaMemcpy(h_v.data() + i + 1, mem.ptr,
                               SizeOf<T>(mem.count), cudaMemcpyDeviceToHost);
      handleStatus(status);

      cublas->copyHostToKernel(v, h_v.data());

      T tau = h_taus[i];
      cublas->copyHostToKernel(Aux1, h_zeros.data());
      cublas->syr(FillMode::FULL, Aux1, &tau, valueType, v);

      cublas->subtract(Aux2, I, Aux1);
      if (i > 0) {
        cublas->matrixMul(Aux1, I, H);
        cublas->matrixMul(H, Aux1, Aux2);
      } else {
        cublas->matrixMul(H, I, Aux2);
      }
    }

    cublas->matrixMul(q[j], I, H);
#if 0
    auto status = cudaMemcpy(q[j]->ptr, H->ptr, SizeOf<T>(H->count),
                             cudaMemcpyDeviceToDevice);
    handleStatus(status);
#endif

    T alpha = static_cast<T>(1);
    T beta = static_cast<T>(0);
    cublas->gemm(r[j], &alpha, valueType, Operation::OP_T, q[j],
                 Operation::OP_N, a[j], &beta, valueType);
  }

  for (const auto *tau : taus) {
    cublas->destroy(tau);
  }

  cublas->destroy(v);
  cublas->destroy(Aux2);
  cublas->destroy(Aux1);
  cublas->destroy(H);
  cublas->destroy(I);
}

void Cublas::_qrDecomposition(Mem *q, Mem *r, Mem *a) {
  Mems qs{q};
  Mems rs{r};
  Mems as{a};

  _qrDecomposition(qs, rs, as);
}

void Cublas::_qrDecomposition(Mems &q, Mems &r, Mems &a) {
  auto valueType = a[0]->valueType;
  if (valueType == ValueType::FLOAT) {
    cublas_qrDecomposition<float>(this, q, r, a, valueType);
  } else if (valueType == ValueType::DOUBLE) {
    cublas_qrDecomposition<double>(this, q, r, a, valueType);
  }
}

void Cublas::_shiftQRIteration(Mem *H, Mem *Q) {
  bool status = false;

  const auto dims = getDims(H);
  const auto _dims = getDims(H);

  auto rows = dims.first;
  auto columns = dims.second;

  const auto valueType = H->valueType;
  const auto _valueType = Q->valueType;

  checkIfAllEqual(valueType, _valueType, "H", "Q");
  checkIfAllEqual(dims, _dims, "H", "Q");

  Mem *aux_Q = createIdentityMatrix(rows, columns, valueType);
  Mem *aux_Q1 = createMatrix(rows, columns, valueType);

  Mem *aux_R = createMatrix(rows, columns, valueType);

  Mem *ioQ = Q;

  status = isUpperTriangular(H);

  while (status == false) {
    qrDecomposition(ioQ, aux_R, H);

    matrixMul(H, aux_R, ioQ);
    matrixMul(aux_Q1, ioQ, aux_Q);
    swap(&aux_Q1, &aux_Q);
    status = isUpperTriangular(H);
  }
}

void cublas_isUpperTriangular(bool &result, Cublas *cublas, Mem *matrix,
                              size_t lda) {
  auto rows = cublas->getRows(matrix);
  auto columns = cublas->getColumns(matrix);

  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  auto factorType = matrix->valueType;
  CudaAlloc alloc;
  Kernels kernels(0, &alloc);

  switch (factorType) {
  case ValueType::FLOAT:
    result = kernels.isUpperTriangular(
        rows, columns, reinterpret_cast<float *>(matrix->ptr), lda, 0.f);
    break;
  case ValueType::DOUBLE:
    result = kernels.isUpperTriangular(
        rows, columns, reinterpret_cast<double *>(matrix->ptr), lda, 0.);
    break;
  case ValueType::FLOAT_COMPLEX:
    result = kernels.isUpperTriangular(
        rows, columns, reinterpret_cast<cuComplex *>(matrix->ptr), lda,
        cuComplex());
    break;
  case ValueType::DOUBLE_COMPLEX:
    result = kernels.isUpperTriangular(
        rows, columns, reinterpret_cast<cuDoubleComplex *>(matrix->ptr), lda,
        cuDoubleComplex());
    break;
  case ValueType::NOT_DEFINED:
    throw std::runtime_error("Not defined value type");
  };
}

void cublas_isUpperTriangular(bool &result, Cublas *cublas, Mem *matrix) {

  auto rows = cublas->getRows(matrix);
  auto columns = cublas->getColumns(matrix);
  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  auto lda = rows;
  cublas_isUpperTriangular(result, cublas, matrix, lda);
}

bool Cublas::_isUpperTriangular(Mem *m) {
  bool result = false;
  cublas_isUpperTriangular(result, this, m);
  return result;
}

void cublas_isLowerTriangular(bool &result, Cublas *cublas, Mem *matrix,
                              size_t lda) {
  auto rows = cublas->getRows(matrix);
  auto columns = cublas->getColumns(matrix);

  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  auto factorType = matrix->valueType;
  CudaAlloc alloc;
  Kernels kernels(0, &alloc);

  switch (factorType) {
  case ValueType::FLOAT:
    result = kernels.isLowerTriangular(
        rows, columns, reinterpret_cast<float *>(matrix->ptr), lda, 0.f);
    break;
  case ValueType::DOUBLE:
    result = kernels.isLowerTriangular(
        rows, columns, reinterpret_cast<double *>(matrix->ptr), lda, 0.);
    break;
  case ValueType::FLOAT_COMPLEX:
    result = kernels.isLowerTriangular(
        rows, columns, reinterpret_cast<cuComplex *>(matrix->ptr), lda,
        cuComplex());
    break;
  case ValueType::DOUBLE_COMPLEX:
    result = kernels.isLowerTriangular(
        rows, columns, reinterpret_cast<cuDoubleComplex *>(matrix->ptr), lda,
        cuDoubleComplex());
    break;
  case ValueType::NOT_DEFINED:
    throw std::runtime_error("Not defined value type");
  };
}

void cublas_isLowerTriangular(bool &result, Cublas *cublas, Mem *matrix) {
  auto rows = cublas->getRows(matrix);
  auto columns = cublas->getColumns(matrix);

  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  auto lda = rows;
  cublas_isLowerTriangular(result, cublas, matrix, lda);
}

bool Cublas::_isLowerTriangular(Mem *m) {
  bool result = false;
  cublas_isLowerTriangular(result, this, m);
  return result;
}

void Cublas::_geam(Mem *output, Mem *alpha, Operation transa, Mem *a, Mem *beta,
                   Operation transb, Mem *b) {
  _geam(output, alpha->ptr, alpha->valueType, transa, a, beta->ptr,
        beta->valueType, transb, b);
}

void Cublas::_geam(Mem *output, void *alpha, ValueType alphaType,
                   Operation transa, Mem *a, void *beta, ValueType betaType,
                   Operation transb, Mem *b) {
  auto m = getRows(a);
  auto n = getColumns(b);

  auto lda = m;
  auto ldb = getRows(b);
  auto ldc = getRows(output);

  auto _cublasSgeam = [this, m, n, lda, ldb, ldc, transa,
                       transb](Mem *output, void *alpha, Mem *a, void *beta,
                               Mem *b) {
    auto *output_f = static_cast<float *>(output->ptr);
    auto *alpha_f = static_cast<float *>(alpha);
    auto *a_f = static_cast<float *>(a->ptr);
    auto *beta_f = static_cast<float *>(beta);
    auto *b_f = static_cast<float *>(b->ptr);
    auto status =
        cublasSgeam(m_handle, convert(transa), convert(transb), m, n, alpha_f,
                    a_f, lda, beta_f, b_f, ldb, output_f, ldc);
    return status;
  };

  auto _cublasDgeam = [this, m, n, lda, ldb, ldc, transa,
                       transb](Mem *output, void *alpha, Mem *a, void *beta,
                               Mem *b) {
    auto *output_f = static_cast<double *>(output->ptr);
    auto *alpha_f = static_cast<double *>(alpha);
    auto *a_f = static_cast<double *>(a->ptr);
    auto *beta_f = static_cast<double *>(beta);
    auto *b_f = static_cast<double *>(b->ptr);
    auto status =
        cublasDgeam(m_handle, convert(transa), convert(transb), m, n, alpha_f,
                    a_f, lda, beta_f, b_f, ldb, output_f, ldc);
    return status;
  };

  using Impl =
      std::function<cublasStatus_t(Mem *, void *, Mem *, void *, Mem *)>;
  const std::map<
      std::tuple<ValueType, ValueType, ValueType, ValueType, ValueType>, Impl>
      impls = {
          {std::make_tuple(ValueType::FLOAT, ValueType::FLOAT, ValueType::FLOAT,
                           ValueType::FLOAT, ValueType::FLOAT),
           std::move(_cublasSgeam)},
          {std::make_tuple(ValueType::DOUBLE, ValueType::DOUBLE,
                           ValueType::DOUBLE, ValueType::DOUBLE,
                           ValueType::DOUBLE),
           std::move(_cublasDgeam)}};

  auto args = std::make_tuple(output->valueType, alphaType, a->valueType,
                              betaType, b->valueType);
  const auto it = impls.find(args);

  if (it == impls.end()) {
    std::stringstream sstream;
    sstream << "Not supported overload: (";
    sstream << toString(std::get<0>(args)) << ", ";
    sstream << toString(std::get<1>(args)) << ", ";
    sstream << toString(std::get<2>(args)) << ", ";
    sstream << toString(std::get<3>(args)) << ", ";
    sstream << toString(std::get<4>(args)) << ")";
    throw std::runtime_error(sstream.str());
  }

  auto status = it->second(output, alpha, a, beta, b);
  handleStatus(status);
}

void Cublas::_add(Mem *output, Mem *a, Mem *b) {
  float oneF = 1.f;
  double oneD = 1.;

  void *one = &oneF;
  ValueType oneType = ValueType::FLOAT;

  if (output->valueType == ValueType::DOUBLE) {
    one = &oneD;
    oneType = ValueType::DOUBLE;
  }

  _geam(output, one, oneType, Operation::OP_N, a, one, oneType, Operation::OP_N,
        b);
}

void Cublas::_subtract(Mem *output, Mem *a, Mem *b) {
  float oneF = 1.f;
  double oneD = 1.;

  float minusOneF = -1.f;
  double minusOneD = -1.;

  void *one = &oneF;
  void *minusOne = &minusOneF;
  ValueType oneType = ValueType::FLOAT;

  if (output->valueType == ValueType::DOUBLE) {
    one = &oneD;
    minusOne = &minusOneD;
    oneType = ValueType::DOUBLE;
  }

  _geam(output, one, oneType, Operation::OP_N, a, minusOne, oneType,
        Operation::OP_N, b);
}

template <typename T>
cublasStatus_t cublas_scaleDiagonal(Cublas *cublas, Mem *matrix, void *factor,
                                    ValueType factorType) {

  auto rows = cublas->getRows(matrix);
  auto columns = cublas->getColumns(matrix);
  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  CudaAlloc cudaAlloc;
  Kernels kernels(0, &cudaAlloc);

  switch (factorType) {
  case ValueType::FLOAT:
    kernels.scaleDiagonal(rows, reinterpret_cast<float *>(matrix->ptr), rows,
                          *reinterpret_cast<float *>(factor));
    break;
  case ValueType::DOUBLE:
    kernels.scaleDiagonal(rows, reinterpret_cast<double *>(matrix->ptr), rows,
                          *reinterpret_cast<double *>(factor));
    break;
  case ValueType::FLOAT_COMPLEX:
    kernels.scaleDiagonal(rows, reinterpret_cast<cuComplex *>(matrix->ptr),
                          rows, *reinterpret_cast<cuComplex *>(factor));
    break;
  case ValueType::DOUBLE_COMPLEX:
    kernels.scaleDiagonal(rows,
                          reinterpret_cast<cuDoubleComplex *>(matrix->ptr),
                          rows, *reinterpret_cast<cuDoubleComplex *>(factor));
    break;
  case ValueType::NOT_DEFINED:
    throw std::runtime_error("Not defined value type");
  };
  return CUBLAS_STATUS_SUCCESS;
}

void Cublas::_scaleDiagonal(Mem *matrix, Mem *factor) {
  _scaleDiagonal(matrix, factor->ptr, factor->valueType);
}

void Cublas::_scaleDiagonal(Mem *matrix, void *factor, ValueType factorType) {
  if (matrix->valueType != factorType) {
    throw std::runtime_error("None identical types");
  }

  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;

  switch (factorType) {
  case ValueType::FLOAT:
    status = cublas_scaleDiagonal<float>(this, matrix, factor, factorType);
    break;
  case ValueType::DOUBLE:
    status = cublas_scaleDiagonal<double>(this, matrix, factor, factorType);
    break;
  case ValueType::FLOAT_COMPLEX:
    status = cublas_scaleDiagonal<cuComplex>(this, matrix, factor, factorType);
    break;
  case ValueType::DOUBLE_COMPLEX:
    status =
        cublas_scaleDiagonal<cuDoubleComplex>(this, matrix, factor, factorType);
    break;
  case ValueType::NOT_DEFINED:
    throw std::runtime_error("Not defined value type");
  };
  handleStatus(status);
}

void Cublas::_tpttr(FillMode uplo, int n, Mem *AP, Mem *A, int lda) {
  auto rows = getRows(A);
  auto columns = getColumns(A);
  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  if (AP->valueType != A->valueType) {
    throw std::runtime_error("Matrices don't have the same value types");
  }

  auto _cublasStpttr = [this](FillMode uplo, int n, Mem *AP, Mem *A, int lda) {
    auto *_AP = static_cast<float *>(AP->ptr);
    auto *_A = static_cast<float *>(A->ptr);
    return cublasStpttr(m_handle, convert(uplo), n, _AP, _A, lda);
  };

  auto _cublasDtpttr = [this](FillMode uplo, int n, Mem *AP, Mem *A, int lda) {
    auto *_AP = static_cast<double *>(AP->ptr);
    auto *_A = static_cast<double *>(A->ptr);
    return cublasDtpttr(m_handle, convert(uplo), n, _AP, _A, lda);
  };

  auto _cublasCtpttr = [this](FillMode uplo, int n, Mem *AP, Mem *A, int lda) {
    auto *_AP = static_cast<cuComplex *>(AP->ptr);
    auto *_A = static_cast<cuComplex *>(A->ptr);
    return cublasCtpttr(m_handle, convert(uplo), n, _AP, _A, lda);
  };

  auto _cublasZtpttr = [this](FillMode uplo, int n, Mem *AP, Mem *A, int lda) {
    auto *_AP = static_cast<cuDoubleComplex *>(AP->ptr);
    auto *_A = static_cast<cuDoubleComplex *>(A->ptr);
    return cublasZtpttr(m_handle, convert(uplo), n, _AP, _A, lda);
  };

  using Impl = std::function<cublasStatus_t(FillMode, int, Mem *, Mem *, int)>;
  const std::map<ValueType, Impl> impls = {
      {ValueType::FLOAT, std::move(_cublasStpttr)},
      {ValueType::DOUBLE, std::move(_cublasDtpttr)},
      {ValueType::FLOAT_COMPLEX, std::move(_cublasCtpttr)},
      {ValueType::DOUBLE_COMPLEX, std::move(_cublasZtpttr)},
  };

  const auto valueType = A->valueType;
  auto it = impls.find(valueType);
  if (it == impls.end()) {
    std::stringstream sstream;
    sstream << "Cannot find implementation for " << toString(valueType);
    throw std::runtime_error(sstream.str());
  }

  auto status = it->second(uplo, n, AP, A, lda);
  handleStatus(status);
}

void Cublas::_trttp(FillMode uplo, int n, Mem *A, int lda, Mem *AP) {
  auto rows = getRows(A);
  auto columns = getColumns(A);
  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  if (AP->valueType != A->valueType) {
    throw std::runtime_error("Matrices don't have the same value types");
  }

  auto _cublasStrttp = [this](FillMode uplo, int n, Mem *A, int lda, Mem *AP) {
    auto *_AP = static_cast<float *>(AP->ptr);
    auto *_A = static_cast<float *>(A->ptr);
    return cublasStrttp(m_handle, convert(uplo), n, _A, lda, _AP);
  };

  auto _cublasDtrttp = [this](FillMode uplo, int n, Mem *A, int lda, Mem *AP) {
    auto *_AP = static_cast<double *>(AP->ptr);
    auto *_A = static_cast<double *>(A->ptr);
    return cublasDtrttp(m_handle, convert(uplo), n, _A, lda, _AP);
  };

  auto _cublasCtrttp = [this](FillMode uplo, int n, Mem *A, int lda, Mem *AP) {
    auto *_AP = static_cast<cuComplex *>(AP->ptr);
    auto *_A = static_cast<cuComplex *>(A->ptr);
    return cublasCtrttp(m_handle, convert(uplo), n, _A, lda, _AP);
  };

  auto _cublasZtrttp = [this](FillMode uplo, int n, Mem *A, int lda, Mem *AP) {
    auto *_AP = static_cast<cuDoubleComplex *>(AP->ptr);
    auto *_A = static_cast<cuDoubleComplex *>(A->ptr);
    return cublasZtrttp(m_handle, convert(uplo), n, _A, lda, _AP);
  };

  using Impl = std::function<cublasStatus_t(FillMode, int, Mem *, int, Mem *)>;
  const std::map<ValueType, Impl> impls = {
      {ValueType::FLOAT, std::move(_cublasStrttp)},
      {ValueType::DOUBLE, std::move(_cublasDtrttp)},
      {ValueType::FLOAT_COMPLEX, std::move(_cublasCtrttp)},
      {ValueType::DOUBLE_COMPLEX, std::move(_cublasZtrttp)},
  };

  const auto valueType = A->valueType;
  auto it = impls.find(valueType);
  if (it == impls.end()) {
    std::stringstream sstream;
    sstream << "Cannot find implementation for " << toString(valueType);
    throw std::runtime_error(sstream.str());
  }

  auto status = it->second(uplo, n, A, lda, AP);
  handleStatus(status);
}

template <typename T> std::string cublas_toString(Cublas *cublas, Mem *mem) {
  std::vector<T> vec;
  vec.resize(mem->count);

  cublas->copyKernelToHost(vec.data(), mem);
  std::stringstream sstream;

  sstream << "[";
  for (int idx = 0; idx < mem->count; ++idx) {
    sstream << vec[idx];
    if (idx < mem->count - 1) {
      sstream << ", ";
    }
  }

  sstream << "]";
  return sstream.str();
}

std::string Cublas::_toStr(Mem *mem) {
  if (mem->valueType == ValueType::FLOAT) {
    return cublas_toString<float>(this, mem);
  } else if (mem->valueType == ValueType::DOUBLE) {
    return cublas_toString<double>(this, mem);
  }
  throw std::runtime_error("Not supported type");
  return "";
}

void Cublas::swap(Mem **a, Mem **b) {
  Mem *temp = *a;
  *a = *b;
  *b = temp;
}

} // namespace mtrx
