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
#include "../dim3_parser.hpp"
#include "../thread_id_parser.hpp"
#include "../uint3_parser.hpp"
#include "dim3_utils.hpp"
#include <mutex>
#include <sstream>
#include <thread>

#include <spdlog/spdlog.h>

namespace mtrx {

void ResetCudaCtx() {
  // blockIdx.clear();
  // blockDim.clear();
  // gridDim.clear();
}

ThreadIdx::ThreadIdxs ThreadIdx::s_threadIdxs;
std::mutex ThreadIdx::s_threadIdxsMutex;

void ThreadIdx::CleanupThreads() { ThreadIdx::s_threadIdxs.clear(); }

ThreadIdx::ThreadIdx() {
  set(m_threadIdx, 0, 0, 0);
  set(m_blockIdx, 0, 0, 0);
  set(m_blockDim, 0, 0, 1);
  set(m_gridDim, 0, 0, 1);
  spdlog::debug("ThreadIdx {} ctor", fmt::ptr(this));
}

ThreadIdx::~ThreadIdx() { spdlog::debug("ThreadIdx {} dtor", fmt::ptr(this)); }

void ThreadIdx::clear() { ::clear(m_threadIdx); }

void ThreadIdx::setThreadIdx(const uint3 &tidx) {
  spdlog::debug("TIDX {} Thread {} {} {}", fmt::ptr(this),
                std::this_thread::get_id(), __func__, tidx);

  m_threadIdx = tidx;
}

void ThreadIdx::setBlockIdx(const dim3 &dim3) {
  spdlog::debug("TIDX {} Thread {} {} {}", fmt::ptr(this),
                std::this_thread::get_id(), __func__, dim3);
  m_blockIdx = dim3;
}

void ThreadIdx::setBlockDim(const dim3 &dim3) {
  spdlog::debug("TIDX {} Thread {} {} {}", fmt::ptr(this),
                std::this_thread::get_id(), __func__, dim3);
  m_blockDim = dim3;
}

void ThreadIdx::setGridDim(const dim3 &dim3) {
  spdlog::debug("TIDX {} Thread {} {} {}", fmt::ptr(this),
                std::this_thread::get_id(), __func__, dim3);
  m_gridDim = dim3;
}

void ThreadIdx::setSharedBuffer(void *buffer) {
  spdlog::debug("TIDX {} Thread {} {} {}", fmt::ptr(this),
                std::this_thread::get_id(), __func__, fmt::ptr(buffer));
  m_sharedBuffer = buffer;
}

const uint3 &ThreadIdx::getThreadIdx() const { return m_threadIdx; }

const dim3 &ThreadIdx::getBlockIdx() const { return m_blockIdx; }

const dim3 &ThreadIdx::getBlockDim() const { return m_blockDim; }

const dim3 &ThreadIdx::getGridDim() const { return m_gridDim; }

void *ThreadIdx::getSharedBuffer() const {
  spdlog::debug("TIDX {} Thread {} {} {}", fmt::ptr(this),
                std::this_thread::get_id(), __func__, fmt::ptr(m_sharedBuffer));
  return m_sharedBuffer;
}

void ThreadIdx::createBarrier(const std::vector<std::thread::id> &threads) {
  std::lock_guard<std::mutex> lg(s_barriersMutex);
  BarrierMutex *bm = nullptr;
  for (size_t tidx = 0; tidx < threads.size(); ++tidx) {
    if (tidx == 0) {
      bm = new BarrierMutex();
      bm->m_barrier.init(threads.size());
    }
    s_barriers[threads[tidx]] = bm;
  }
}

void ThreadIdx::destroyBarrier(const std::vector<std::thread::id> &threads) {
  std::lock_guard<std::mutex> lg(s_barriersMutex);
  for (size_t fa = 0; fa < threads.size(); ++fa) {
    if (fa == 0) {
      delete s_barriers[threads[fa]];
    }
    s_barriers.erase(threads[fa]);
  }
}

void ThreadIdx::wait() {
  const auto tid = std::this_thread::get_id();
  auto it = s_barriers.find(tid);
  if (it == s_barriers.end()) {
    std::stringstream sstream;
    sstream << "Thread " << tid << " is not registered for barrier";
    throw std::runtime_error(sstream.str());
  }
  spdlog::debug("Thread {} waits on barrier {}", std::this_thread::get_id(),
                fmt::ptr(&it->second->m_barrier));
  it->second->m_barrier.wait();
  spdlog::debug("Thread {} unlocked from a wait on barrier {}",
                std::this_thread::get_id(), fmt::ptr(&it->second->m_barrier));
}

ThreadIdx::Barriers ThreadIdx::s_barriers;
std::mutex ThreadIdx::s_barriersMutex;
} // namespace mtrx
