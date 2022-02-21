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

#ifndef MTRX_CUBLAS_DEVICE_PROPERTIES_LOADER_H
#define MTRX_CUBLAS_DEVICE_PROPERTIES_LOADER_H

#include <cuda.h>
#include <vector>

#include "device_properties.hpp"

namespace mtrx {

void loadDeviceProperties(DeviceProperties &deviceProperties, CUdevice device);
DeviceProperties loadDeviceProperties(CUdevice device);

void loadAttributes(std::vector<int> &output,
                    const std::vector<CUdevice_attribute> &attributes,
                    CUdevice device);

} // namespace mtrx

#endif
