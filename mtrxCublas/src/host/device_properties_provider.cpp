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

#include "device_properties_provider.hpp"
#include <sstream>

namespace mtrx {
std::map<int, DeviceProperties> DevicePropertiesProvider::m_deviceProperties;

void DevicePropertiesProvider::set(int device,
                                   const DeviceProperties &deviceProperties) {
  m_deviceProperties[device] = deviceProperties;
}

DeviceProperties DevicePropertiesProvider::get(int device) {
  auto it = m_deviceProperties.find(device);
  if (it == m_deviceProperties.end()) {
    std::stringstream sstream;
    sstream << "Not existed device under index " << device;
    throw std::runtime_error(sstream.str());
  }
  return it->second;
}

} // namespace mtrx
