/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "device_info.hpp"
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace mtrx {
DeviceInfo::DeviceInfo() : m_cuDevice(0) {}

DeviceInfo::~DeviceInfo() { m_cuDevice = 0; }

CUdevice DeviceInfo::getDevice() const { return m_cuDevice; }

void DeviceInfo::getDeviceProperties(DeviceProperties &devProp) {
  initDeviceProperties();

  std::memcpy(&devProp, &m_deviceProperties, sizeof(DeviceProperties));
}

uint DeviceInfo::getMaxThreadsX() const {
  checkInitialization();
  return m_deviceProperties.maxThreadsCount[0];
}

uint DeviceInfo::getMaxThreadsY() const {
  checkInitialization();
  return m_deviceProperties.maxThreadsCount[1];
}

uint DeviceInfo::getMaxBlocksX() const {
  checkInitialization();
  return m_deviceProperties.maxBlocksCount[0];
}

uint DeviceInfo::getMaxBlocksY() const {
  checkInitialization();
  return m_deviceProperties.maxBlocksCount[1];
}

uint DeviceInfo::getSharedMemorySize() const {
  checkInitialization();
  return m_deviceProperties.sharedMemPerBlock;
}

uint DeviceInfo::getMaxThreadsPerBlock() const {
  checkInitialization();
  return m_deviceProperties.maxThreadsPerBlock;
}

void DeviceInfo::setDevice(CUdevice cuDevice) {
  this->m_cuDevice = cuDevice;
  initDeviceProperties();
}

void DeviceInfo::initDeviceProperties() {
  if (!m_initialized) {
    for (size_t idx = 0; idx < 9; ++idx) {
      int result;
      /*printCuError */ (
          cuDeviceGetAttribute(&result, m_attributes[idx], m_cuDevice));
      m_values[idx] = result;
    }

    std::memcpy(&m_deviceProperties, m_values, sizeof(m_values));
    m_initialized = true;
  }
}

std::string DeviceInfo::toStr() {
  initDeviceProperties();
  std::stringstream sstream;

  sstream << "Device properties:" << std::endl;
  sstream << "--Max grid size: " << m_deviceProperties.maxBlocksCount[0] << ", "
          << m_deviceProperties.maxBlocksCount[1] << ", "
          << m_deviceProperties.maxBlocksCount[2] << std::endl;
  sstream << "--Max threads dim: " << m_deviceProperties.maxThreadsCount[0]
          << ", " << m_deviceProperties.maxThreadsCount[1] << ", "
          << m_deviceProperties.maxThreadsCount[2] << std::endl;
  sstream << "--Max threads per block: "
          << m_deviceProperties.maxThreadsPerBlock << std::endl;
  sstream << "--Max registers per block: "
          << m_deviceProperties.maxRegistersPerBlock << std::endl;
  sstream << "--Shared memory per block in bytes: "
          << m_deviceProperties.sharedMemPerBlock << std::endl;

  return sstream.str();
}

void DeviceInfo::checkInitialization() const {
  if (!m_initialized) {
    throw std::runtime_error("Cuda context is not initialized");
  }
}
} // namespace mtrx
