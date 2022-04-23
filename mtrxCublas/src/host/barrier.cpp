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

#include "barrier.hpp"
#include "../thread_id_parser.hpp"
#include <spdlog/spdlog.h>
#include <thread>

namespace mtrx {

Barrier::Barrier(unsigned int count) : m_init(false) { this->init(count); }

void Barrier::init(unsigned int count) {
  if (m_init) {
    pthread_barrier_destroy(&m_barrier);
  }

  pthread_barrier_init(&m_barrier, NULL, count);

  m_init = true;
}

void Barrier::wait() {
  spdlog::debug("Barrier: Thread {} waits on barrier {}",
                std::this_thread::get_id(), fmt::ptr(this));
  pthread_barrier_wait(&m_barrier);
  spdlog::debug("Barrier: Thread {} released from barrier {}",
                std::this_thread::get_id(), fmt::ptr(this));
}
} // namespace mtrx
