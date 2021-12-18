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

#include "thread_idx.hpp"
#include "dim3_utils.hpp"
#include <sstream>

namespace mtrx {

void ResetCudaCtx() {
  // blockIdx.clear();
  // blockDim.clear();
  // gridDim.clear();
}

ThreadIdx::ThreadIdxs ThreadIdx::m_threadIdxs;
std::mutex ThreadIdx::m_threadIdxsMutex;

void ThreadIdx::CleanupThreads()
{
  ThreadIdx::m_threadIdxs.clear();
}

ThreadIdx::ThreadIdx()
{
  set(m_threadIdx, 0, 0, 0);
  set(m_blockIdx, 0, 0, 0);
  set(m_blockDim, 0, 0, 1);
  set(m_gridDim, 0, 0, 1);
}

void ThreadIdx::clear() { ::clear(m_threadIdx); }

void ThreadIdx::setThreadIdx(const uint3 &tidx) { m_threadIdx = tidx; }

void ThreadIdx::setBlockIdx(const dim3 &dim3) { m_blockIdx = dim3; }

void ThreadIdx::setBlockDim(const dim3 &dim3) { m_blockDim = dim3; }

void ThreadIdx::setGridDim(const dim3 &dim3) { m_gridDim = dim3; }

void ThreadIdx::setSharedBuffer(void *buffer) { m_sharedBuffer = buffer; }

const uint3 &ThreadIdx::getThreadIdx() const { return m_threadIdx; }

const dim3 &ThreadIdx::getBlockIdx() const { return m_blockIdx; }

const dim3 &ThreadIdx::getBlockDim() const { return m_blockDim; }

const dim3 &ThreadIdx::getGridDim() const { return m_gridDim; }

void *ThreadIdx::getSharedBuffer() const { return m_sharedBuffer; }

void ThreadIdx::createBarrier(const std::vector<std::thread::id> &threads) {
  m_barriersMutex.lock();
  BarrierMutex *bm = NULL;
  for (size_t fa = 0; fa < threads.size(); ++fa) {
    if (fa == 0) {
      bm = new BarrierMutex();
      bm->m_barrier.init(threads.size());
    }
    m_barriers[threads[fa]] = bm;
  }
  m_barriersMutex.unlock();
}

void ThreadIdx::destroyBarrier(const std::vector<std::thread::id> &threads) {
  m_barriersMutex.lock();
  for (size_t fa = 0; fa < threads.size(); ++fa) {
    if (fa == 0) {
      delete m_barriers[threads[fa]];
    }
    m_barriers.erase(threads[fa]);
  }
  m_barriersMutex.unlock();
}

void ThreadIdx::wait() {
  const auto tid = std::this_thread::get_id();
  auto it = m_barriers.find(tid);
  if (it == m_barriers.end()) {
    std::stringstream sstream;
    sstream << "Thread " << tid << " is not in barrier";
    throw std::runtime_error(sstream.str());
  }
  it->second->m_barrier.wait();
}

ThreadIdx::Barriers ThreadIdx::m_barriers;
std::mutex ThreadIdx::m_barriersMutex;
} // namespace mtrx
