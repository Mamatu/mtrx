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

#ifndef MTRX_CORE_MEM_PTR_HPP
#define MTRX_CORE_MEM_PTR_HPP

#include <functional>
#include <memory>
#include <mtrxCore/types.hpp>

namespace mtrx {
using mem_unique = std::unique_ptr<Mem, std::function<void(Mem *)>>;
using mem_shared = std::shared_ptr<Mem>;

class MemUnique : public mem_unique {
public:
  MemUnique(MemUnique &&memUnique) : mem_unique(std::move(memUnique)) {}
};

class MemShared : public mem_shared {
public:
  MemShared(const MemShared &memShared) : mem_shared(memShared) {}

  MemShared(MemShared &&memShared) : mem_shared(std::move(memShared)) {}

  MemShared(MemUnique &&memUnique) : mem_shared(std::move(memUnique)) {}
};
} // namespace mtrx

#endif
