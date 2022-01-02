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

#include "ikernel_executor.hpp"

#include <spdlog/spdlog.h>
#include <sstream>

namespace mtrx {

std::string toString(const RunParams &runParams) {
  std::stringstream sstream;
  const uint *tc = runParams.threadsCount;
  sstream << "threadsCount " << tc[0] << ", " << tc[1] << ", " << tc[2]
          << std::endl;
  const uint *bc = runParams.blocksCount;
  sstream << "blocksCount " << bc[0] << ", " << bc[1] << ", " << bc[2]
          << std::endl;
  sstream << "sharedMemory " << runParams.sharedMemSize << std::endl;

  return sstream.str();
}

IKernelExecutor::IKernelExecutor() { reset(); }

IKernelExecutor::~IKernelExecutor() {}

void IKernelExecutor::setRunParams(const RunParams &runParams) {
  m_runParams = runParams;
}

const RunParams &IKernelExecutor::getRunParams() const { return m_runParams; }

void IKernelExecutor::setBlocksCount(uint x, uint y, uint z) {
  m_runParams.blocksCount[0] = x;
  m_runParams.blocksCount[1] = y;
  m_runParams.blocksCount[2] = z;
}

void IKernelExecutor::setThreadsCount(uint x, uint y, uint z) {
  m_runParams.threadsCount[0] = x;
  m_runParams.threadsCount[1] = y;
  m_runParams.threadsCount[2] = z;
}

void IKernelExecutor::setSharedMemory(uint size) {
  m_runParams.sharedMemSize = size;
}

void IKernelExecutor::setParams(const void **params) { m_params = params; }

int IKernelExecutor::getParamsCount() const { return m_paramsCount; }

bool IKernelExecutor::run(const std::string &function) {

  auto params_str = toString(getRunParams());
  spdlog::info("{} {} '{}'\n{}", __FILE__, __func__, function, params_str);

  bool status = _run(function);
  reset();
  return status;
}

bool IKernelExecutor::run(const std::string &function, const void **params) {
  setParams(params);
  return run(function);
}

bool IKernelExecutor::run(const std::string &function, const void **params,
                          uint sharedMemorySize) {
  m_runParams.sharedMemSize = sharedMemorySize;
  return run(function, params);
}

const void **IKernelExecutor::getParams() const { return m_params; }

void IKernelExecutor::reset() {
  m_runParams.blocksCount[0] = 1;
  m_runParams.blocksCount[1] = 1;
  m_runParams.blocksCount[2] = 1;

  m_runParams.threadsCount[0] = 1;
  m_runParams.threadsCount[1] = 1;
  m_runParams.threadsCount[2] = 1;

  m_runParams.sharedMemSize = 0;

  m_params = nullptr;
  m_paramsCount = 0;
}
} // namespace mtrx
