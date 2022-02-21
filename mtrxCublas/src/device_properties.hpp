/*
 * Copyright 2016 - 2021 Marcin Matula
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

#ifndef MTRX_CUBLAS_DEVICE_PROPERTIES_H
#define MTRX_CUBLAS_DEVICE_PROPERTIES_H

#include <array>

namespace mtrx {
struct DeviceProperties {
  std::array<int, 3> blockDim;
  std::array<int, 3> gridDim;

  int maxThreadsPerBlock;
  int maxRegistersPerBlock;
  int sharedMemPerBlock;
};
} // namespace mtrx

#endif
