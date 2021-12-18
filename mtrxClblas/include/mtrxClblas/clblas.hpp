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

#ifndef MTRX_CLBLAS_HPP
#define MTRX_CLBLAS_HPP

#include <mtrxCore/blas.hpp>
#include <mtrxCore/types.hpp>

namespace mtrx {
class Clblas : public Blas {
public:
  Clblas();
  virtual ~Clblas() override;

protected:
  Mem *_createMem(size_t size, ValueType valueType) override;
  void _destroy(const Mem *mem) override;

  uintt _getCount(const Mem *mem) const override;

  void _copyHostToKernel(Mem *mem, void *array) override;
  void _copyKernelToHost(void *array, Mem *mem) override;

  uintt _amax(const Mem *mem) override;
  uintt _amin(const Mem *mem) override;

  void _rot(Mem *x, Mem *y, void *c, ValueType cType, void *s,
            ValueType sType) override;
  void _rot(Mem *x, Mem *y, Mem *c, Mem *s) override;

private:
  // cublasHandle_t m_handle;
};
} // namespace mtrx

#endif
