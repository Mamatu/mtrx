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

#include "host_kernel_thread.hpp"
#include "thread_idx.hpp"
#include <mutex>
#include <spdlog/spdlog.h>
#include <thread>

#include "../thread_id_parser.hpp"

namespace mtrx {

HostKernelThread::HostKernelThread() : m_pthreads(nullptr), m_barrier(nullptr) {
  m_cancontinue = false;
  spdlog::debug("HKT {} ctor", fmt::ptr(this));
}

void HostKernelThread::setThreadFunc(ThreadFunc &&threadFunc) {
  m_threadFunc = std::move(threadFunc);
}

void HostKernelThread::setThreadFunc(const ThreadFunc &threadFunc) {
  m_threadFunc = threadFunc;
}
void HostKernelThread::setBlockDim(const dim3 &blockDim) {
  m_blockDim = blockDim;
}

dim3 HostKernelThread::getThreadIdx() const { return m_threadIdx; }

dim3 HostKernelThread::getBlockIdx() const { return m_blockIdx; }

void HostKernelThread::setGridDim(const dim3 &gridDim) { m_gridDim = gridDim; }

void HostKernelThread::setThreadIdx(const dim3 &threadIdx) {
  m_threadIdx = threadIdx;
}

void HostKernelThread::setBlockIdx(const dim3 &blockIdx) {
  m_blockIdx = blockIdx;
}

void HostKernelThread::setPthreads(std::vector<std::thread::id> *pthreads) {
  m_pthreads = pthreads;
}

void HostKernelThread::setBarrier(mtrx::Barrier *barrier) {
  m_barrier = barrier;
}

void HostKernelThread::setSharedBuffer(void *buffer) {
  spdlog::debug("HKT {} {} {}", fmt::ptr(this), __func__, fmt::ptr(buffer));
  m_sharedBuffer = buffer;
}

HostKernelThread::~HostKernelThread() {
  if (m_thread && m_thread->joinable()) {
    m_thread->join();
  }
  spdlog::debug("HKT {} dtor", fmt::ptr(this));
}

void HostKernelThread::waitOn() {
  std::unique_lock<std::mutex> lock(m_mutex);
  if (m_cancontinue == false) {
    spdlog::debug("HKT {} Thread {} waits", fmt::ptr(this), m_thread->get_id());
    m_cond.wait(lock);
    spdlog::debug("HKT {} Thread {} unlocked from a wait", fmt::ptr(this),
                  m_thread->get_id());
  }
  m_cancontinue = false;
}

void HostKernelThread::run() {
  m_cancontinue = false;
  m_thread = std::make_shared<std::thread>(HostKernelThread::Execute, this);
  spdlog::debug("HKT {} Thread created: {}", fmt::ptr(this),
                m_thread->get_id());
  {
    std::unique_lock<std::mutex> ul(m_mutex);
    m_cond.wait(ul, [this]() { return m_isPrepared; });
  }
}

std::thread::id HostKernelThread::get_id() const {
  return std::this_thread::get_id();
}

void HostKernelThread::Execute(HostKernelThread *threadImpl) {
  threadImpl->initInThread();

  spdlog::debug("{}:{} HKT {} Thread {} is executed", __func__, __LINE__,
                fmt::ptr(threadImpl), std::this_thread::get_id());
  const int gridSize = threadImpl->m_gridDim.x * threadImpl->m_gridDim.y;
  for (int gridIdx = 0; gridIdx < gridSize; ++gridIdx) {
    threadImpl->m_barrier->wait();

    ThreadIdx &ti = ThreadIdx::GetThreadIdx();
    ti.setThreadIdx(threadImpl->m_threadIdx);
    ti.setBlockIdx(threadImpl->m_blockIdx);

    threadImpl->m_threadFunc(threadImpl->m_threadIdx, threadImpl->m_blockIdx);

    {
      std::unique_lock<std::mutex> locker(threadImpl->m_mutex);
      threadImpl->m_cancontinue = true;
    }
    threadImpl->m_cond.notify_one();
  }
}

void HostKernelThread::initInThread() {
  std::thread::id threadId = std::this_thread::get_id();
  ThreadIdx &threadIndex = ThreadIdx::GetThreadIdx();
  threadIndex.setThreadIdx(m_threadIdx);
  threadIndex.setBlockIdx(m_blockIdx);
  threadIndex.setBlockDim(m_blockDim);
  threadIndex.setGridDim(m_gridDim);
  threadIndex.setSharedBuffer(m_sharedBuffer);

  m_pthreads->push_back(threadId);

  if (m_threadIdx.x == m_blockDim.x - 1 && m_threadIdx.y == m_blockDim.y - 1) {
    ThreadIdx::createBarrier(*m_pthreads);
  }

  {
    std::lock_guard<std::mutex> lg(m_mutex);
    m_isPrepared = true;
  }

  m_cond.notify_one();
}
} // namespace mtrx
