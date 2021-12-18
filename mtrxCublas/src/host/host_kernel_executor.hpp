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

#ifndef MTRX_CUBLAS_HOST_KERNEL_EXECUTOR_H
#define MTRX_CUBLAS_HOST_KERNEL_EXECUTOR_H

//#include "dim3.hpp"
#include "../ikernel_executor.hpp"

namespace mtrx {

class HostKernelExecutor : public IKernelExecutor {
public:
  HostKernelExecutor(uint maxThreadsPerBlocks = 1024);

  virtual ~HostKernelExecutor();

protected:
  bool _run(const std::string &functionName) override;

private:
  uint m_maxThreadsPerBlock;
};

} // namespace mtrx

#endif
