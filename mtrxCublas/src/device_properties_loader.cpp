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

#include "device_properties_loader.hpp"

#include <mtrxCublas/status_handler.hpp>

#include <array>

namespace mtrx {

template <typename Attributes>
void _loadAttributes(std::vector<int> &output, const Attributes &attributes,
                     CUdevice device) {
  for (size_t idx = 0; idx < attributes.size(); ++idx) {
    int result = -1;
    CUresult curesult = cuDeviceGetAttribute(&result, attributes[idx], device);
    handleStatus(curesult);
    output.push_back(result);
  }
}

void loadDeviceProperties(DeviceProperties &deviceProperties, CUdevice device) {
  const std::array<CUdevice_attribute, 9> m_attributes = {
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK};

  std::vector<int> output;
  _loadAttributes(output, m_attributes, device);

  std::memcpy(&deviceProperties, output.data(), output.size() * sizeof(int));
}

DeviceProperties loadDeviceProperties(CUdevice device) {
  DeviceProperties deviceProperties;
  loadDeviceProperties(deviceProperties, device);
  return deviceProperties;
}

void loadAttributes(std::vector<int> &output,
                    const std::vector<CUdevice_attribute> &attributes,
                    CUdevice device) {
  _loadAttributes(output, attributes, device);
}

} // namespace mtrx
