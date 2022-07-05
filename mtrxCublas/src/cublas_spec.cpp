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
#include "cublas.hpp"
#include "cublas_v2.h"
#include "driver_types.h"
#include "mtrxCore/status_handler.hpp"
#include "mtrxCore/types.hpp"
#include <array>
#include <cassert>
#include <cstdio>
#include <exception>

#include <cuda_runtime.h>
#include <mtrxCore/checkers.hpp>

#include <map>
#include <mtrxCore/size_of.hpp>
#include <mtrxCublas/status_handler.hpp>
#include <mtrxCublas/to_string.hpp>
#include <spdlog/fmt/bundled/format-inl.h>
#include <spdlog/spdlog.h>
#include <type_traits>
#include <utility>

#include "cuda_alloc.hpp"
#include "kernels.hpp"

#ifdef CUBLAS_NVPROF_KERNELS
#include "cuda_profiler.hpp"
#define PROFILER() Profiler p;
#else
#define PROFILER()
#endif

namespace mtrx {

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

CublasSpec::CublasSpec() { handleStatus(cublasCreate(&m_handle)); }

CublasSpec::~CublasSpec() {
  try {
    handleStatus(cublasDestroy(m_handle));
  } catch (const std::exception &ex) {
    spdlog::error("%s", ex.what());
    abort();
  }
}

std::vector<int> CublasSpec::getDevices() const {
  int count = 0;
  handleStatus(cudaGetDeviceCount(&count));

  std::vector<int> devices;
  devices.reserve(count);

  for (int i = 0; i < count; ++i) {
    devices.push_back(i);
  }

  return devices;
}

void CublasSpec::setDevice(int device) { handleStatus(cudaSetDevice(device)); }

void CublasSpec::dealloc(float *mem) { cublas_dealloc(mem); }

void CublasSpec::dealloc(double *mem) { cublas_dealloc(mem); }

void CublasSpec::dealloc(cuComplex *mem) { cublas_dealloc(mem); }

void CublasSpec::dealloc(cuDoubleComplex *mem) { cublas_dealloc(mem); }

uintt CublasSpec::_getCount(const Mem *mem) const { return mem->count; }

uintt CublasSpec::_getSizeInBytes(const Mem *mem) const {
  return SizeOf(mem->valueType, mem->count);
}

void CublasSpec::_copyHostToKernel(float *mem, float *array) {
  auto status =
      cublasSetVector(mem->count, SizeOf<float>(), array, 1, mem->ptr, 1);
  handleStatus(status);
}

void CublasSpec::_copyHostToKernel(Mem *mem, const void *array) {
  auto status = cublasSetVector(mem->count, SizeOf(mem->valueType), array, 1,
                                mem->ptr, 1);
  handleStatus(status);
}

void CublasSpec::_copyKernelToHost(void *array, Mem *mem) {
  auto status = cublasGetVector(mem->count, SizeOf(mem->valueType), mem->ptr, 1,
                                array, 1);
  handleStatus(status);
}

template <typename H, typename T, typename CublasIamax>
uintt cublas_amam(Hr &handler, const T *mem, CublasSpec &cublas,
                  CublasIamax &&cublasIamam) {
  const uintt n = cublas.getCount(mem);
  int resultIdx = -1;
  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
  status = cublasIamam(handle, n, static_cast<T *>(mem), 1, &resultIdx);
  handleStatus(status);

  if (resultIdx == -1) {
    throw std::runtime_error("resultIdx is -1");
  }

  return resultIdx - 1;
}

uintt CublasSpec::_amax(const float *mem) {
  return cublas_amam(m_handle, mem, *this, cublasIsamax);
}

uintt CublasSpec::_amax(const double *mem) {
  return cublas_amam(m_handle, mem, *this, cublasIdamax);
}

uintt CublasSpec::_amax(const cuComplex *mem) {
  return cublas_amam(m_handle, mem, *this, cublasIcamax);
}

uintt CublasSpec::_amax(const cuDoubleComplex *mem) {
  return cublas_amam(m_handle, mem, *this, cublasIzamax);
}

uintt CublasSpec::_amin(const float *mem) {
  return cublas_amam(m_handle, mem, *this, cublasIsamin);
}

uintt CublasSpec::_amin(const double *mem) {
  return cublas_amam(m_handle, mem, *this, cublasIdamin);
}

uintt CublasSpec::_amin(const cuComplex *mem) {
  return cublas_amam(m_handle, mem, *this, cublasIcamin);
}

uintt CublasSpec::_amin(const cuDoubleComplex *mem) {
  return cublas_amam(m_handle, mem, *this, cublasIzamin);
}

template <typename H, typename T, CublasRot>
void cublas_rot(H handle, T *x, T *y, T *c, T *s, CublasSpec &cublas,
                CublasRot &&cublasrot) {
  auto x_count = cublas.getCount(x);
  auto y_count = cublas.getCount(y);
  if (x_count != y_count) {
    std::stringstream sstream;
    sstream << "Different count for x and y. x: " << x->count
            << " y: " << y->count;
    throw std::runtime_error(sstream.str());
  }

  const auto n = x_count;

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

  auto status = cublasrot(handle, n, x, 1, y, 1, c, s);
  handleStatus(status);
}

void CublasSpec::_rot(float *x, float *y, float *c, float *s) {
  cublas_rot(m_handle, x, y, c, s, *this, cublasSrot);
}

void CublasSpec::_rot(double *x, double *y, double *c, double *s) {
  cublas_rot(m_handle, x, y, c, s, *this, cublasDrot);
}

template <typename H, typename T, typename CublasSyr>
void cublas_syr(H handle, FillMode fillMode, T *output, T *alpha, T *x,
                CublasSyr &&cublassyr) {
  int n = getRows(output);
  int m = getColumns(output);
  const int lda = n;

  if (n != m) {
    std::stringstream sstream;
    sstream << "Matrix 'output' must be square not " << n << "x" << m;
    throw std::runtime_error(sstream.str());
  }

  auto call = [&handle, it, output, alpha, x](FillMode fillMode) {
    auto status = cublassyr(handle, fillMode, output, alpha, x);
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

void CublasSpec::_syr(FillMode fillMode, float *output, float alpha, float *x) {
  cublas_syr(m_handle, fillMode, output, &alpha, x, cublasSsyr);
}

void CublasSpec::_syr(FillMode fillMode, double *output, double alpha,
                      double *x) {
  cublas_syr(m_handle, fillMode, output, &alpha, x, cublasDsyr);
}

template <typename H, typename T, typename CublasGemm>
void cublas_gemm(H handle, T *output, T *alpha, Operation transa, T *a,
                 Operation transb, T *b, T *beta, CublasGemm &&cublasgemm) {
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

  auto *alpha_f = static_cast<double *>(alpha);
  auto *output_f = static_cast<double *>(output->ptr);
  auto *a_f = static_cast<double *>(a->ptr);
  auto *b_f = static_cast<double *>(b->ptr);
  auto *beta_f = static_cast<double *>(beta);
  auto status = cublasgemm(m_handle, cutransa, cutransb, m, n, k, alpha_f, a_f,
                           m, b_f, k, beta_f, output_f, m);
  handleStatus(status);
}

void CublasSpec::_gemm(float *output, float *alpha, Operation transa, float *a,
                       Operation transb, float *b, float *beta) {
  cublas_gemm(m_handle, output, alpha, transa, a, transb, b, beta, *this);
}

void CublasSpec::_gemm(double *output, double *alpha, Operation transa,
                       double *a, Operation transb, double *b, double *beta) {
  cublas_gemm(m_handle, output, alpha, transa, a, transb, b, beta, *this);
}

void CublasSpec::_symm(SideMode sideMode, FillMode fillMode, Mem *output,
                       void *alpha, ValueType alphaType, Mem *a, Mem *b,
                       void *beta, ValueType betaType) {
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

void CublasSpec::_symm(SideMode sideMode, FillMode fillMode, Mem *output,
                       Mem *alpha, Mem *a, Mem *b, Mem *beta) {
  _symm(sideMode, fillMode, output, alpha->ptr, alpha->valueType, a, b,
        beta->ptr, beta->valueType);
}

template <typename H, typename T, typename CublasgeqrfBatched>
void cublas_geqrf(H handle, CublasSpec::Vec<T> &a, CublasSpec::Vec<T> &tau,
                  CublasgeqrfBatched &&cublasgeqrfBatched) {
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

  auto create = [this, &checkDim](auto &array, const auto &input) {
    int m = -1;
    int n = -1;

    array.reserve(input.size());

    int _rows = -1;
    int _columns = -1;

    for (const auto *elem : input) {
      array.push_back(elem->ptr);

      auto rows = getRows(elem);
      auto columns = getColumns(elem);
      auto valueType = elem->valueType;

      checkDim(_rows, _columns, rows, columns);

      _rows = rows;
      _columns = columns;
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

  int info = 0;

  static_assert(sizeof(float *) == sizeof(void *),
                "Mismatch beetwen size of float* and void*");

  auto **a_t = reinterpret_cast<T **>(a.data());
  auto **tau_t = reinterpret_cast<T **>(tau.data());

  auto **d_a = cublas_allocCudaArrayCopy<T *>(a_f, a.size());
  auto **d_tau = cublas_allocCudaArrayCopy<T *>(tau_f, tau.size());

  auto status =
      cublasgeqrfBatched(handle, m, n, d_a, lda, d_tau, &info, batchSize);

  cublas_deallocCudaArray(d_a);
  cublas_deallocCudaArray(d_tau);

  handleStatus(status);

  if (info < 0) {
    std::stringstream sstream;
    sstream << "The parameters passed at " << -info << " is invalid";
    throw std::runtime_error(sstream.str());
  }
}

void CublasSpec::_geqrf(Vec<float> &a, Vec<float> &tau) {
  cublas_geqrf(m_handle, a, tau, cublasSgeqrfBatched);
}

void CublasSpec::_geqrf(Vec<double> &a, Vec<double> &tau) {
  cublas_geqrf(m_handle, a, tau, cublasSgeqrfBatched);
}

template <typename T>
void cublas_qrDecomposition(CublasSpec *cublas, CublasSpec::Vec<T> &q,
                            CublasSpec::Vec<T> &r, CublasSpec::Vec<T> &a) {
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

  CublasSpec::Vec<T> aux3;
  for (auto *mema : a) {
    Mem *Aux3 = cublas->createMatrix(m, m, valueType);
    cublas->matrixMul(Aux3, I, mema);
    aux3.push_back(Aux3);
  }

  auto dim = std::max(static_cast<size_t>(1), std::min(m, n));

  CublasSpec::Vec<T> taus;
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

void CublasSpec::_qrDecomposition(Vec<float> &q, Vec<float> &r, Vec<float> &a) {
  cublas_qrDecomposition(this, q, r, a);
}

void CublasSpec::_qrDecomposition(Vec<double> &q, Vec<double> &r,
                                  Vec<double> &a) {
  cublas_qrDecomposition(this, q, r, a);
}

template <typename T> void cublas_shiftQRIteration(T *H, T *Q) {
  bool status = false;

  const auto dims = getDims(H);
  const auto _dims = getDims(H);

  auto rows = dims.first;
  auto columns = dims.second;

  const auto valueType = H->valueType;
  const auto _valueType = Q->valueType;

  checkIfAllEqual(valueType, _valueType, "H", "Q");
  checkIfAllEqual(dims, _dims, "H", "Q");

  T *aux_Q = createIdentityMatrix(rows, columns);
  T *aux_Q1 = createMatrix(rows, columns);

  T *aux_R = createMatrix(rows, columns);

  T *ioQ = Q;

  status = isUpperTriangular(H);

  while (status == false) {
    qrDecomposition(ioQ, aux_R, H);

    matrixMul(H, aux_R, ioQ);
    matrixMul(aux_Q1, ioQ, aux_Q);
    swap(&aux_Q1, &aux_Q);
    status = isUpperTriangular(H);
  }
}

void CublasSpec::_shiftQRIteration(float *H, float *Q) {
  cublas_shiftQRIteration(H, Q);
}

void CublasSpec::_shiftQRIteration(double *H, double *Q) {
  cublas_shiftQRIteration(H, Q);
}

void CublasSpec::_geam(Mem *output, Mem *alpha, Operation transa, Mem *a,
                       Mem *beta, Operation transb, Mem *b) {
  _geam(output, alpha->ptr, alpha->valueType, transa, a, beta->ptr,
        beta->valueType, transb, b);
}

void CublasSpec::_geam(Mem *output, void *alpha, ValueType alphaType,
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

void CublasSpec::_add(Mem *output, Mem *a, Mem *b) {
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

void CublasSpec::_subtract(Mem *output, Mem *a, Mem *b) {
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
cublasStatus_t cublas_scaleDiagonal(CublasSpec *cublas, Mem *matrix,
                                    void *factor, ValueType factorType) {

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

void CublasSpec::_scaleDiagonal(Mem *matrix, Mem *factor) {
  _scaleDiagonal(matrix, factor->ptr, factor->valueType);
}

void CublasSpec::_scaleDiagonal(Mem *matrix, void *factor,
                                ValueType factorType) {
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

void CublasSpec::_tpttr(FillMode uplo, int n, Mem *AP, Mem *A, int lda) {
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

void CublasSpec::_trttp(FillMode uplo, int n, Mem *A, int lda, Mem *AP) {
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

template <typename T>
std::string cublas_toString(CublasSpec *cublas, Mem *mem) {
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

bool CublasSpec::_isUnit(Mem *mem, void *delta, ValueType deltaType) {
  CudaAlloc alloc;
  Kernels kernels(0, &alloc);

  auto m = getRows(mem);
  auto n = getColumns(mem);
  auto lda = m;

  auto _isUnitS = [m, n, lda, &kernels](Mem *mem, void *delta) {
    auto *mem_c = static_cast<float *>(mem->ptr);
    auto *delta_c = static_cast<float *>(delta);
    return kernels.isUnit(m, n, mem_c, lda, *delta_c);
  };

  auto _isUnitD = [m, n, lda, &kernels](Mem *mem, void *delta) {
    auto *mem_c = static_cast<double *>(mem->ptr);
    auto *delta_c = static_cast<double *>(delta);
    return kernels.isUnit(m, n, mem_c, lda, *delta_c);
  };

  auto _isUnitC = [m, n, lda, &kernels](Mem *mem, void *delta) {
    auto *mem_c = static_cast<cuComplex *>(mem->ptr);
    auto *delta_c = static_cast<float *>(delta);
    return kernels.isUnit(m, n, mem_c, lda, *delta_c);
  };

  auto _isUnitZ = [m, n, lda, &kernels](Mem *mem, void *delta) {
    auto *mem_c = static_cast<cuDoubleComplex *>(mem->ptr);
    auto *delta_c = static_cast<double *>(delta);
    return kernels.isUnit(m, n, mem_c, lda, *delta_c);
  };

  using Impl = std::function<bool(Mem * mem, void *delta)>;
  const std::map<std::pair<ValueType, ValueType>, Impl> impls = {
      {{ValueType::FLOAT, ValueType::FLOAT}, std::move(_isUnitS)},
      {{ValueType::DOUBLE, ValueType::DOUBLE}, std::move(_isUnitD)},
      {{ValueType::FLOAT_COMPLEX, ValueType::FLOAT}, std::move(_isUnitC)},
      {{ValueType::DOUBLE_COMPLEX, ValueType::DOUBLE}, std::move(_isUnitZ)},
  };

  auto it = impls.find(std::make_pair(mem->valueType, deltaType));

  if (it == impls.end()) {
    std::stringstream sstream;
    sstream << "Cannot find implementation for " << toString(mem->valueType)
            << " " << toString(deltaType);
    throw std::runtime_error(sstream.str());
  }

  return it->second(mem, delta);
}

bool CublasSpec::_isUnit(Mem *mem, Mem *delta) {
  return _isUnit(m, n, mem, lda, delta->ptr, delta->valueType);
}

std::string CublasSpec::_toStr(Mem *mem) {
  if (mem->valueType == ValueType::FLOAT) {
    return cublas_toString<float>(this, mem);
  } else if (mem->valueType == ValueType::DOUBLE) {
    return cublas_toString<double>(this, mem);
  }
  throw std::runtime_error("Not supported type");
  return "";
}

void CublasSpec::swap(Mem **a, Mem **b) {
  Mem *temp = *a;
  *a = *b;
  *b = temp;
}

} // namespace mtrx
