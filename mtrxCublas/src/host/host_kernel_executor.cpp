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

#include "host_kernel_executor.hpp"
#include "host_kernel.hpp"

#include "cuda_kernels_list.hpp"

#include <functional>
#include <map>
#include <sstream>

#include <spdlog/spdlog.h>

namespace mtrx {

std::map<std::string, std::function<void(const void **)>> g_kernelsList = {
    {"CUDAKernel_SF_scaleTrace", proxy_HOSTKernel_SF_scaleTrace},
    {"CUDAKernel_SD_scaleTrace", proxy_HOSTKernel_SD_scaleTrace},
    {"CUDAKernel_CF_scaleTrace", proxy_HOSTKernel_CF_scaleTrace},
    {"CUDAKernel_CD_scaleTrace", proxy_HOSTKernel_CD_scaleTrace}};

class HostKernelImpl : public HostKernel {
  std::function<void(const void **)> m_function;
  const void **m_params;

public:
  HostKernelImpl(const std::function<void(const void **)> &function,
                 const void **params, void *ctx)
      : HostKernel(ctx, false), m_function(function), m_params(params) {}

protected:
  void execute(const dim3 &threadIdx, const dim3 &blockIdx) {
    m_function(m_params);
  }
};

HostKernelExecutor::HostKernelExecutor(uint maxThreadsPerBlock)
    : m_maxThreadsPerBlock(maxThreadsPerBlock) {}

HostKernelExecutor::~HostKernelExecutor() { HostKernel::ReleaseThreads(this); }

bool HostKernelExecutor::_run(const std::string &functionName) {
  auto it = g_kernelsList.find(functionName);

  if (it == g_kernelsList.end()) {
    std::stringstream sstr;
    sstr << "Function name ";
    sstr << functionName;
    sstr << " is not registered in g_kernelsList";
    throw std::runtime_error(sstr.str());
    return false;
  }

  const mtrx::RunParams &params = this->getRunParams();
 
  auto params_str = toString(params);
  SPDLOG_INFO("{} {}\n{}", __FILE__, __func__, params_str);

  HostKernelImpl hki(it->second, getParams(), this);

  dim3 blockDim(params.threadsCount[0], params.threadsCount[1], params.threadsCount[2]);
  dim3 gridDim(params.blocksCount[0], params.blocksCount[1], params.blocksCount[2]);

  hki.setDims(gridDim, blockDim);
  hki.setSharedMemory(params.sharedMemSize);

  hki.executeKernelAsync();

  return true;
}
} // namespace mtrx
