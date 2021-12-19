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

#ifndef MTRX_CUBLAS_IKERNEL_EXECUTOR_H
#define MTRX_CUBLAS_IKERNEL_EXECUTOR_H

#include <string>

namespace mtrx {
struct RunParams {
  uint threadsCount[3];
  uint blocksCount[3];
  uint sharedMemSize;
};

std::string toString(const RunParams &runParams);

class IKernelExecutor {
public:
  IKernelExecutor();
  virtual ~IKernelExecutor();

  void setRunParams(const RunParams &runParams);
  const RunParams &getRunParams() const;

  void setBlocksCount(uint x, uint y, uint z);
  void setThreadsCount(uint x, uint y, uint z);
  void setSharedMemory(uint size);

  void setParams(const void **params);
  const void **getParams() const;
  int getParamsCount() const;

  bool run(const std::string &function);
  bool run(const std::string &function, const void **params);
  bool run(const std::string &function, const void **params,
           uint sharedMemorySize);

protected:
  virtual bool _run(const std::string &function) = 0;

private:
  RunParams m_runParams;

  const void **m_params = nullptr;
  int m_paramsCount = 0;

  void reset();
};
} // namespace mtrx

#endif
