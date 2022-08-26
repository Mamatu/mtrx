
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

#ifndef MTRX_IRAM_IRAM_ROUTINES_HPP
#define MTRX_IRAM_IRAM_ROUTINES_HPP

#include <mtrxCore/blas.hpp>
#include <mtrxCore/types.hpp>
#include <mtrxCublas/cublas_types.hpp>
#include <mtrxIram/iram_types.hpp>

#include <memory>

namespace mtrx {

using BlasPtr = std::shared_ptr<mtrx::Blas>;
void createRandomUnitVector(const BlasPtr &blasPtr, int length,
                            ValueType valueType);
void createUnitVector(const BlasPtr &blasPtr, int length, ValueType valueType);
void checkCustomInitVector(const BlasPtr &blasPtr, Mem *mem);

} // namespace mtrx

#endif
