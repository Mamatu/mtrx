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

#ifndef MTRX_CUBLAS_IKERNEL_EXECUTOR_H
#define MTRX_CUBLAS_IKERNEL_EXECUTOR_H

#include <array>
#include <string>

#include "device_properties.hpp"

#include <cuComplex.h>

namespace mtrx {
struct RunParams {
  uint threadsCount[3];
  uint blocksCount[3];
  uint sharedMemSize;
};

std::string toString(const RunParams &runParams);

class IKernelExecutor {
public:
  IKernelExecutor(const DeviceProperties &deviceProperties);
  virtual ~IKernelExecutor();

  void setRunParams(const RunParams &runParams);
  const RunParams &getRunParams() const;

  void setBlocksCount(uint x, uint y, uint z);
  void setThreadsCount(uint x, uint y, uint z);

  void setBlocksCount(const dim3 &xy) {
    setBlocksCount(xy.x, xy.y, xy.z);
  }

  void setThreadsCount(const dim3 &xy) {
    setThreadsCount(xy.x, xy.y, xy.z);
  }

  void setSharedMemory(uint size);

  void setParams(const void **params);
  const void **getParams() const;
  int getParamsCount() const;

  bool run(const std::string &function);
  bool run(const std::string &function, const void **params);
  bool run(const std::string &function, const void **params,
           uint sharedMemorySize);

  DeviceProperties getDeviceProperties() const;

protected:
  virtual bool _run(const std::string &function) = 0;

private:
  DeviceProperties m_deviceProperties;
  RunParams m_runParams;

  const void **m_params = nullptr;
  int m_paramsCount = 0;

  void reset();
};
} // namespace mtrx

#endif
