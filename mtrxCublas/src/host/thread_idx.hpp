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

#ifndef MTRX_THREAD_IDX_HPP
#define MTRX_THREAD_IDX_HPP

#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "barrier.hpp"
#include <vector_types.h>

namespace mtrx {

class ThreadIdx final {
public:
  ThreadIdx();
  ~ThreadIdx() = default;

  static void CleanupThreads();

  using ThreadIdxs = std::map<std::thread::id, ThreadIdx>;
  template <typename T, typename Callback>
  static T GetThreadIdxsSafe(Callback &&callback) {
    std::lock_guard<std::mutex> lg(ThreadIdx::m_threadIdxsMutex);
    return callback(ThreadIdx::m_threadIdxs);
  }

  static ThreadIdx &GetThreadIdx(std::thread::id id) {
    auto getter = [id](auto &threadIdxs) -> ThreadIdx & {
      return ThreadIdx::m_threadIdxs[id];
    };
    return GetThreadIdxsSafe<ThreadIdx &>(std::move(getter));
  }

  static ThreadIdx &GetThreadIdx() {
    return GetThreadIdx(std::this_thread::get_id());
  }

  void setThreadIdx(const uint3 &tidx);
  void setBlockIdx(const dim3 &dim3);
  void setBlockDim(const dim3 &dim3);
  void setGridDim(const dim3 &dim3);
  void setSharedBuffer(void *buffer);

  const uint3 &getThreadIdx() const;
  const dim3 &getBlockIdx() const;
  const dim3 &getBlockDim() const;
  const dim3 &getGridDim() const;
  void *getSharedBuffer() const;

  static void createBarrier(const std::vector<std::thread::id> &threads);
  static void destroyBarrier(const std::vector<std::thread::id> &threads);
  static void wait();

  void clear();

private:
  uint3 m_threadIdx;
  dim3 m_blockIdx;
  dim3 m_blockDim;
  dim3 m_gridDim;
  void *m_sharedBuffer;

  class BarrierMutex {
  public:
    mtrx::Barrier m_barrier;
    std::mutex m_mutex;
  };

  using Barriers = std::map<std::thread::id, BarrierMutex *>;
  static Barriers m_barriers;
  static std::mutex m_barriersMutex;

  static ThreadIdxs m_threadIdxs;
  static std::mutex m_threadIdxsMutex;
};
} // namespace mtrx
#endif /* MTRX_DIM3_HPP */
