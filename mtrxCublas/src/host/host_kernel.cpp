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

#include "host_kernel.hpp"
#include "thread_idx.hpp"
#include "threads_mapper.hpp"

#include <memory>
#include <spdlog/spdlog.h>

#include "../thread_id_parser.hpp"

namespace mtrx {
std::map<void *, HostKernel::ThreadsPool> HostKernel::s_threads;
std::mutex HostKernel::s_mutex;

HostKernel::HostKernel() : m_ctx(this), m_release(true) {}
HostKernel::HostKernel(void *ctx, bool releaseThreads)
    : m_ctx(ctx), m_release(releaseThreads) {}

HostKernel::~HostKernel() {
  if (m_release) {
    ReleaseThreads(m_ctx);
  }
}

void HostKernel::ReleaseThreads(void *ctx) {
  std::lock_guard<std::mutex> lg(s_mutex);
  auto it = s_threads.find(ctx);
  if (it != s_threads.end()) {
    for (auto it1 = it->second.begin(); it1 != it->second.end(); ++it1) {
      delete it1->second;
    }
    it->second.clear();
    s_threads.erase(it);
  }
}

void HostKernel::setDims(const dim3 &gridDim, const dim3 &blockDim) {
  this->blockDim = blockDim;
  this->gridDim = gridDim;
  onSetDims(this->gridDim, this->blockDim);
}

void HostKernel::calculateDims(uintt columns, uintt rows) {
  uint blocks[2];
  uint threads[2];
  mtrx::utils::mapper::SetThreadsBlocks(blocks, threads, columns, rows, 1024);
  setDims(dim3(blocks[0], blocks[1]), dim3(threads[0], threads[1]));
}

void HostKernel::setSharedMemory(size_t sizeInBytes) {
  m_sharedMemorySize = sizeInBytes;
}

void HostKernel::executeKernelAsync() {
  dim3 threadIdx(0, 0, 0);
  dim3 blockIdx(0, 0, 0);
  const dim3 blockDim = this->blockDim;
  const dim3 gridDim = this->gridDim;

  const bool correctDim =
      (gridDim.y != 0 && gridDim.x != 0 && blockDim.y != 0 && blockDim.x != 0);
  if (!correctDim) {
    throw std::runtime_error("Invalid dim");
  }
  std::vector<mtrx::HostKernelThread *> m_threads;
  std::vector<std::thread::id> m_pthreads;

  unsigned int count = blockDim.y * blockDim.x + 1;

  spdlog::debug("Creates barrier with {} count ({} * {} + 1)", count, blockDim.y, blockDim.x);
  mtrx::Barrier barrier(count);
  m_pthreads.reserve(blockDim.x * blockDim.y);

  for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY) {
    for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX) {
      threadIdx.x = threadIdxX;
      threadIdx.y = threadIdxY;
      if (blockIdx.x == 0 && blockIdx.y == 0) {
        {
          std::lock_guard<std::mutex> lg(s_mutex);
          auto it = s_threads.find(m_ctx);
          if (it == s_threads.end()) {
            s_threads[m_ctx] = ThreadsPool();
          }
        }
        {
          std::lock_guard<std::mutex> lg(s_mutex);
          auto it =
              s_threads[m_ctx].find(std::make_pair(threadIdxX, threadIdxY));
          if (it == s_threads[m_ctx].end()) {
            mtrx::HostKernelThread *threadImpl = new mtrx::HostKernelThread();
            s_threads[m_ctx][std::make_pair(threadIdxX, threadIdxY)] =
                threadImpl;
          }
        }

        mtrx::HostKernelThread *threadImpl =
            s_threads[m_ctx][std::make_pair(threadIdxX, threadIdxY)];

        threadImpl->setThreadFunc([this](dim3 threadIdx, dim3 blockIdx) {
          execute(threadIdx, blockIdx);
          onChange(HostKernel::CUDA_THREAD, threadIdx, blockIdx);
        });

        threadImpl->setBlockDim(blockDim);
        threadImpl->setGridDim(gridDim);
        threadImpl->setThreadIdx(threadIdx);
        threadImpl->setBlockIdx(blockIdx);
        threadImpl->setPthreads(&m_pthreads);
        threadImpl->setBarrier(&barrier);
        m_threads.push_back(threadImpl);
      }
    }
  }

  std::map<std::pair<int, int>, std::vector<char>> sharedMemories;
  std::unique_ptr<char[]> sharedMemory(nullptr);
  spdlog::debug("Runs kernel in gridDim {} {} {}", gridDim.x, gridDim.y,
                gridDim.z);
  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
      if (m_sharedMemorySize > 0) {
        std::vector<char> sharedMemory;
        sharedMemory.resize(m_sharedMemorySize);
        spdlog::debug("Allocates shared memory at {} with size {}",
                      fmt::ptr(sharedMemory.data()), m_sharedMemorySize);
        sharedMemories[std::make_pair(blockIdxX, blockIdxY)] = sharedMemory;
      }

      blockIdx.x = blockIdxX;
      blockIdx.y = blockIdxY;

      for (size_t tidx = 0; tidx < m_threads.size(); ++tidx) {
        m_threads.at(tidx)->setBlockIdx(blockIdx);
        m_threads.at(tidx)->setSharedBuffer(
            sharedMemories[std::make_pair(blockIdxX, blockIdxY)].data());
        if (blockIdx.x == 0 && blockIdx.y == 0) {
          spdlog::debug("Runs thread with idx {}", tidx);
          m_threads.at(tidx)->run();
        }
      }

      spdlog::debug("Thread {} waits on barrier {}", std::this_thread::get_id(), fmt::ptr(&barrier));
      barrier.wait();
      spdlog::debug("Thread {} unlocked from a wait on barrier {}", std::this_thread::get_id(), fmt::ptr(&barrier));

      for (size_t tidx = 0; tidx < m_threads.size(); ++tidx) {
        m_threads.at(tidx)->waitOn();
      }

      this->onChange(HostKernel::CUDA_BLOCK, threadIdx, blockIdx);
    }
  }

  m_threads.clear();
  ThreadIdx::destroyBarrier(m_pthreads);
  ThreadIdx::CleanupThreads();
  m_pthreads.clear();
}

void HostKernel::executeKernelSync() {
  dim3 threadIdx;
  dim3 blockIdx;
  const dim3 gridDim = this->gridDim;
  const dim3 blockDim = this->blockDim;

  const bool validDim =
      (gridDim.y != 0 && gridDim.x != 0 && blockDim.y != 0 && blockDim.x != 0);
  if (!validDim) {
    throw std::runtime_error("Invalid dim");
  }

  for (uintt blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
    for (uintt blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
      std::unique_ptr<char[]> sharedMemory(nullptr);
      if (m_sharedMemorySize > 0) {
        sharedMemory.reset(new char[m_sharedMemorySize]);
      }
      for (uintt threadIdxY = 0; threadIdxY < blockDim.y; ++threadIdxY) {
        for (uintt threadIdxX = 0; threadIdxX < blockDim.x; ++threadIdxX) {
          threadIdx.x = threadIdxX;
          threadIdx.y = threadIdxY;
          blockIdx.x = blockIdxX;
          blockIdx.y = blockIdxY;
          ThreadIdx &ti = ThreadIdx::GetThreadIdx();
          ti.setThreadIdx(threadIdx);
          ti.setBlockIdx(blockIdx);
          ti.setBlockDim(blockDim);
          ti.setGridDim(gridDim);
          ti.setSharedBuffer(sharedMemory.get());
          this->execute(threadIdx, blockIdx);
          this->onChange(HostKernel::CUDA_THREAD, threadIdx, blockIdx);
        }
      }
      this->onChange(HostKernel::CUDA_BLOCK, threadIdx, blockIdx);
    }
  }
}
} // namespace mtrx
