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

#ifndef MTRX_CUBLAS_DEVICE_INFO_H
#define MTRX_CUBLAS_DEVICE_INFO_H

#include <cuda.h>
#include <string>

namespace mtrx {
struct DeviceProperties {
  int maxThreadsCount[3];
  int maxBlocksCount[3];

  int maxThreadsPerBlock;
  int maxRegistersPerBlock;
  int sharedMemPerBlock;
};

class DeviceInfo {
public:
  DeviceInfo();
  virtual ~DeviceInfo();

  DeviceInfo(const DeviceInfo &) = delete;
  DeviceInfo &operator=(const DeviceInfo &) = delete;

  CUdevice getDevice() const;

  void setDevice(CUdevice cuDecive);

  uint getMaxThreadsX() const;
  uint getMaxThreadsY() const;
  uint getMaxBlocksX() const;
  uint getMaxBlocksY() const;

  uint getSharedMemorySize() const;
  uint getMaxThreadsPerBlock() const;

  std::string toStr();
  bool isInitialized() const { return m_initialized; }
  void getDeviceProperties(DeviceProperties &devProp);

private:
  CUdevice m_cuDevice = 0;

  bool m_initialized = false;

  int m_values[9];
  const CUdevice_attribute m_attributes[9] = {
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK};

  void checkInitialization() const;
  void initDeviceProperties();

protected:
  DeviceProperties m_deviceProperties;
};
} // namespace mtrx

#endif
