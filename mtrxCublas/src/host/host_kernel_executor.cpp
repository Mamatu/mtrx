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
#include "device_properties_provider.hpp"
#include "host_kernel.hpp"

#include "cuda_proxies.hpp"

#include <functional>
#include <map>
#include <sstream>

#include <spdlog/spdlog.h>

namespace mtrx {

std::map<std::string, std::function<void(const void **)>> g_kernelsList = {
    {"CudaKernel_SF_scaleTrace", proxy_HostKernel_SF_scaleTrace},
    {"CudaKernel_SD_scaleTrace", proxy_HostKernel_SD_scaleTrace},
    {"CudaKernel_CF_scaleTrace", proxy_HostKernel_CF_scaleTrace},
    {"CudaKernel_CD_scaleTrace", proxy_HostKernel_CD_scaleTrace},
    {"CudaKernel_SF_isUpperTriangular", proxy_HostKernel_SF_isUpperTriangular},
    {"CudaKernel_SD_isUpperTriangular", proxy_HostKernel_SD_isUpperTriangular},
    {"CudaKernel_CF_isUpperTriangular", proxy_HostKernel_CF_isUpperTriangular},
    {"CudaKernel_CD_isUpperTriangular", proxy_HostKernel_CD_isUpperTriangular},
    {"CudaKernel_SF_isLowerTriangular", proxy_HostKernel_SF_isLowerTriangular},
    {"CudaKernel_SD_isLowerTriangular", proxy_HostKernel_SD_isLowerTriangular},
    {"CudaKernel_CF_isLowerTriangular", proxy_HostKernel_CF_isLowerTriangular},
    {"CudaKernel_CD_isLowerTriangular", proxy_HostKernel_CD_isLowerTriangular},
    {"CudaKernel_SI_reduceShm", proxy_HostKernel_SI_reduceShm},
    {"CudaKernel_SF_reduceShm", proxy_HostKernel_SF_reduceShm},
    {"CudaKernel_SD_reduceShm", proxy_HostKernel_SD_reduceShm},
    {"CudaKernel_CF_reduceShm", proxy_HostKernel_CF_reduceShm},
    {"CudaKernel_CD_reduceShm", proxy_HostKernel_CD_reduceShm},
};

class HostKernelImpl : public HostKernel {
  std::function<void(const void **)> m_function;
  const void **m_params;

public:
  HostKernelImpl(const std::function<void(const void **)> &function,
                 const void **params, void *ctx)
      : HostKernel(ctx, false), m_function(function), m_params(params) {}

protected:
  void execute(const dim3 & /*threadIdx*/, const dim3 & /*blockIdx*/) {
    m_function(m_params);
  }
};

HostKernelExecutor::HostKernelExecutor(int device)
    : IKernelExecutor(DevicePropertiesProvider::get(device)), m_device(device) {
}

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

  dim3 blockDim(params.threadsCount[0], params.threadsCount[1],
                params.threadsCount[2]);
  dim3 gridDim(params.blocksCount[0], params.blocksCount[1],
               params.blocksCount[2]);

  hki.setDims(gridDim, blockDim);
  hki.setSharedMemory(params.sharedMemSize);

  hki.executeKernelAsync();

  return true;
}
} // namespace mtrx
