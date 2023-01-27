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

#ifndef MTRX_IRAM_SHIFTED_QR_ITERATION_HPP
#define MTRX_IRAM_SHIFTED_QR_ITERATION_HPP

#include <mtrxCore/thrower.hpp>
#include <vector>

namespace mtrx {

template <typename Blas, typename T>
void shifted_qr_iteration_check(Blas &&blas, T *A, T *V, T *H) {
  auto h_dims = blas.getDims(H);
  throw_exception_ifnot(h_dims.first == h_dims.second, [&h_dims](auto &in) {
    in << "Matrix H must be square matrix. It is " << h_dims.first << "x" << h_dims.second;
  });

  auto v_dims = blas.getDims(V);
  throw_exception_ifnot(v_dims.first == v_dims.second, [&v_dims](auto &in) {
    in << "Matrix V must be square matrix. It is: " << v_dims.first << "x" << v_dims.second;
  });

  auto a_dims = blas.getDims(A);
  throw_exception_ifnot(a_dims.first == a_dims.second, [&a_dims](auto &in) {
    in << "Matrix A must be square matrix. It is: " << a_dims.first << "x" << a_dims.second;
  });

  throw_exception_ifnot(
      a_dims.first == h_dims.first && a_dims.second == h_dims.second,
      [](auto &in) { in << "Matrices A and H must have the same dimension"; });

  throw_exception_ifnot(
      v_dims.first == h_dims.first && v_dims.second == h_dims.second,
      [](auto &in) { in << "Matrices V and H must have the same dimension"; });

  throw_exception_ifnot(blas.isUpperHessenberg(H),
                        [](auto &in) { in << "H must be upper hessenberg"; });
}

template <typename Blas, typename T>
void shifted_qr_iteration(Blas &&blas, T *A, T *V, T *H, T *Q, T *R, T *H_uI,
                          bool check = true) {
  if (check) {
    shifted_qr_iteration_check(std::forward<Blas>(blas), A, V, H);
  }
  auto columns = blas.getColumns(H);

  std::vector<T> diagonals(columns, 0);
  blas.copyKernelToHost(diagonals.data(), 1, H, columns, diagonals.size());
  for (auto it = diagonals.begin(); it != diagonals.end(); ++it) {
    blas.copyKernelToKernel(H_uI, H);
    const auto &u = *it;
    blas.diagonalAdd(H_uI, -u);
    blas.qrDecomposition(Q, R, H_uI);
  }
}

template <typename Blas, typename T>
void shifted_qr_iteration(Blas &&blas, T *A, T *V, T *H, bool check = true) {
  if (check) {
    shifted_qr_iteration_check(std::forward<Blas>(blas), A, V, H);
  }
  auto rows = blas.getRows(H);
  auto columns = blas.getColumns(H);

  throw_exception_ifnot(rows == columns, [](auto &in) {
    in << "Matrix H must be square matrix";
  });

  auto *H_uI = blas.createMatrix(rows, columns);
  auto *Q = blas.createMatrix(rows, columns);
  auto *R = blas.createMatrix(rows, columns);

  shifted_qr_iteration(std::forward<Blas>(blas), A, V, H, Q, R, H_uI, false);
}

template <typename Blas, typename T> class ShiftedQRIteration final {
public:
  ShiftedQRIteration(Blas &&_blas, T *_A, T *_V, T *_H)
      : blas(std::forward<Blas>(_blas)), A(_A), V(_V), H(_H) {}

  ~ShiftedQRIteration() {
    blas.destroy(Q);
    blas.destroy(R);
    blas.destroy(H_uI);
  }

  ShiftedQRIteration(const ShiftedQRIteration<Blas, T> &orig) = delete;
  ShiftedQRIteration(ShiftedQRIteration<Blas, T> &&orig) = delete;
  ShiftedQRIteration &
  operator=(const ShiftedQRIteration<Blas, T> &orig) = delete;
  ShiftedQRIteration &operator=(ShiftedQRIteration<Blas, T> &&orig) = delete;

  void operator()() {
    shifted_qr_iteration_check(std::forward<Blas>(blas), A, V, H);
    auto rows = blas.getRows(H);
    auto columns = blas.getColumns(H);
    if (H_uI == nullptr) {
      H_uI = blas.createMatrix(rows, columns);
      Q = blas.createMatrix(rows, columns);
      R = blas.createMatrix(rows, columns);
    }
    shifted_qr_iteration(std::forward<Blas>(blas), A, V, H, Q, R, H_uI, false);
  }

private:
  Blas blas;
  T *A = nullptr;
  T *V = nullptr;
  T *H = nullptr;
  T *Q = nullptr;
  T *R = nullptr;
  T *H_uI = nullptr;
};
} // namespace mtrx

#endif
